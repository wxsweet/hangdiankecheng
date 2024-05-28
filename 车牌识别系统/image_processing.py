import os
import cv2
from PIL import Image
import numpy as np
import function
import svm_message
SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
# Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

#车牌识别中的机器学习任务
class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)#核类型（RBF核）
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        #预测输入样本的类别
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")

    def img_first_process(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件，二值图像 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = function.img_read(car_pic_file)   #读取文件
        else:
            img = car_pic_file
        # 优化处理速度和效果，因为过大的图像可能不必要地增加计算负担
        pic_hight, pic_width = img.shape[:2]  #取彩色图片的高、宽
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            # 缩小图片
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # 关于interpolation 有几个参数可以选择:
        # cv2.INTER_AREA - 局部像素重采样，适合缩小图片。
        # cv2.INTER_CUBIC和 cv2.INTER_LINEAR 更适合放大图像，其中INTER_LINEAR为默认方法。

        img = cv2.GaussianBlur(img, (5, 5), 0)
        # 高斯滤波是一种线性平滑滤波，对于除去高斯噪声有很好的效果
        # 0 是指根据窗口大小（ 5,5 ）来计算高斯函数标准差
        

        oldimg = img
        # 转化成灰度图像
        # 转换颜色空间 cv2.cvtColor
        # BGR ---> Gray  cv2.COLOR_BGR2GRAY
        # BGR ---> HSV  cv2.COLOR_BGR2HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("./tmp/img_gray.jpg", img)
        
        #ones()返回一个全1的n维数组 
        Matrix = np.ones((20, 20), np.uint8)  

        # 开运算:先进行腐蚀再进行膨胀就叫做开运算。它被用来去除噪声。 cv2.MORPH_OPEN
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("./tmp/img_opening.jpg", img_opening)
        # 图片叠加与融合
        # g (x) = (1 − α)f0 (x) + αf1 (x)   a→（0，1）不同的a值可以实现不同的效果
        # 图像融合
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        cv2.imwrite("./tmp/img_ronghe.jpg", img_opening)
        # 创建20*20的元素为1的矩阵 开操作，并和img重合


        # Otsu’s二值化
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Canny 边缘检测
        # 较大的阈值2用于检测图像中明显的边缘  一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的
        # 较小的阈值1用于将这些间断的边缘连接起来
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imwrite("./tmp/img_edge.jpg", img_edge)

        Matrix = np.ones((4, 19), np.uint8)
        # 闭运算:先膨胀再腐蚀，闭运算可以填充图像中的小孔，同时也可以连接相邻的物体
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        # 开运算：开运算可以消除图像中的小孔，同时也可以分离相邻的物体。
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("./tmp/img_xingtai.jpg", img_edge2)
        return img_edge2, oldimg
    
    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        img_contours：图像轮廓
        :return: 识别到的字符、定位的车牌图像、车牌颜色
        """
        pic_hight, pic_width = img_contours.shape[:2] # #取彩色图片的高、宽
#三个颜色范围的阈值
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])

        # BGR ---> HSV
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        # 利用cv2.inRange函数设阈值，去除背景部分
        # 参数1：原图
        # 参数2：图像中低于值，图像值变为0
        # 参数3：图像中高于值，图像值变为0
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)

        # 图像算术运算  按位运算 按位操作有： AND， OR， NOT， XOR 等
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 根据阈值找到对应颜色
        cv2.imwrite("./tmp/img_corlor.jpg", output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./tmp/img_huidu.jpg", output)
        Matrix = np.ones((20, 20), np.uint8)
        #使用一个 20x20 的卷积核
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)  #闭运算
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix) #开运算
#使用function.img_findContours函数找到边缘轮廓
        card_contours = function.img_findContours(img_edge2)
        # for rect in card_contours :
        #     box = cv2.boxPoints(rect)  # 获取轮廓的四个角点
        #     box = np.int0(box)  # 转换为整数
        #     cv2.drawContours(oldimg, [box], 0, (0, 255, 0), 2)  # 在原始图像上绘制轮廓
        # cv2.imwrite("./tmp/img_contour.jpg",oldimg)
        # function.img_Transform函数对轮廓进行变换。
        card_imgs = function.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        i=0
        # for item in card_imgs:
        #     i=i+1
        #     cv2.imwrite("./tmp/img_jiaozheng{}.jpg".format(i), item)
        # 使用function.img_color函数获取颜色信息。
        colors, car_imgs = function.img_color(card_imgs)
        # cv2.imwrite("./tmp/img_yansemay.jpg",car_imgs[0])
        # cv2.imwrite("./tmp/img_yansemay1.jpg", car_imgs[1])
        predict_result = []
        predict_str = ""
        roi = None
        card_color = None
        # 遍历颜色列表，找到蓝色、黄色和绿色的颜色，然后对对应的卡片图像进行灰度转换。如果灰度转换失败，会打印
        # "gray转换失败"。
        for i, color in enumerate(colors):

            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                cv2.imwrite("./tmp/img_chepailast.jpg",card_img)
                try:
                    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                except:
                    print("gray转换失败")

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)#二进制位运算
#执行该代码后，ret表示阈值计算的结果，gray_img为经过二值化处理后的图像。
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite("./tmp/img_yansegray.jpg", gray_img)
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = function.find_waves(x_threshold, x_histogram)
                #如果波形峰值数量为0，则跳过当前循环，继续处理下一个车牌图像。
                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                # 这段代码的作用是找到wave_peaks列表中，元素值之差最大的元组。其中，wave_peaks是一个包含多个元组的列表，每个元组有两个元素，表示波形峰值的起始和结束位置。
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])

                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                cv2.imwrite("./tmp/img_yansegray1.jpg", gray_img)
                #计算子图像的垂直直方图，并找到直方图中的波峰。
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = function.find_waves(y_threshold, y_histogram)
                #如果波峰的数量小于6，那么跳过当前循环，继续处理下一个子图像。
                if len(wave_peaks) < 6:
                    # print("peak less 1:", len(wave_peaks))
                    continue
                # 如果波峰的数量大于等于6，那么找到波峰中距离最远的两个波峰，计算它们的距离，
                # 并将结果存储在变量max_wave_dis中。
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                # 通过判断第一个波形的宽度是否小于最大波形距离的三分之一且起始位置为0来判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                # 代码使用一个循环来遍历剩余的波形。在循环中，它计算当前波形与前一个波形之间的距离，
                # 并与最大波形距离的60 % 进行比较。如果当前波形与前一个波形之间的距离大于最大波形距离的60 %，
                # 则跳出循环。否则，将当前波形与前一个波形的距离累加到cur_dis变量中。
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                        # 如果循环中有至少一个波形被处理（即i > 0），
                        # 则将第一个波形与最后一个被处理的波形组合成一个新的波形，并将其插入到列表的开头。
                        # 这样，汉字就被正确地组合在一起了。
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue
                # print(wave_peaks)
                
                # wave_peaks  车牌字符 类型列表 包含7个（开始的横坐标，结束的横坐标）



                part_cards =function .seperate_card(gray_img, wave_peaks)

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        # print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    
                    part_card = svm_message.preprocess_hog([part_card])
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = svm_message.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)

                roi = card_img
                card_color = color
                break
        cv2.imwrite("./tmp/img_caijian.jpg", roi)
        # print(predict_str)
        return predict_str, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色
