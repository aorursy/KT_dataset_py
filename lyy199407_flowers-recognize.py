import cv2

import numpy as np

import os

import time

from sklearn.svm import SVC

from sklearn.externals import joblib

from scipy.cluster.vq import vq,kmeans

from sklearn.preprocessing import StandardScaler



start = time.time()

train_path = "../input/flowers/flowers/train"

training_names = os.listdir(train_path)

image_classes = []

class_path =[]

class_id = 0



#每张图像的路径存储在image_paths列表中

image_paths = []



for training_name in training_names:

    dir = os.path.join(train_path, training_name)

    for i in os.listdir(dir):

        temp = os.path.join(dir,i)

        class_path.append(temp)

    image_paths += class_path

    image_classes += [class_id] * len(class_path)

    class_id += 1

    class_path =[]



# 创建特征提取器

sift = cv2.xfeatures2d.SIFT_create()



# 特征提取与描述子生成

des_list = []



for image_path in image_paths:

    im = cv2.imread(image_path)

    im = cv2.resize(im, (300,300 ))    #将输入图像缩放成统一尺寸

    kpts = sift.detect(im)

    kpts, des = sift.compute(im,kpts)

    des_list.append((image_path, des))



# 描述子向量

descriptors = des_list[0][1]

for image_path, descriptor in des_list[1:]:

    descriptors = np.vstack((descriptors, descriptor))



# 用K-Means算法生成K个基础词汇

k = 600

voc, variance = kmeans(descriptors, k, 1)



# 生成特征直方图

im_features = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):

    words, distance = vq(des_list[i][1], voc)

    for w in words:

        im_features[i][w] += 1



# 归一化特征向量

stdSlr = StandardScaler().fit(im_features)

im_features = stdSlr.transform(im_features)



# 训练SVM

clf = SVC(max_iter=3000)

clf.fit(im_features, np.array(image_classes))



#保存SVM

joblib.dump((clf, training_names, stdSlr, k, voc), "bow.pkl", compress=3)



finish = time.time()



print("模型训练完毕，总共耗时{}s".format(finish-start))
import os

import cv2

import numpy as np

from sklearn.externals import joblib

from scipy.cluster.vq import vq



# 加载分类器，类别名称，尺度，K值，词典

clf, classes_names, stdSlr, k, voc = joblib.load("bow.pkl")

# 创建特征提取器

sift = cv2.xfeatures2d.SIFT_create()





def predict_image(image_path):

    # 将所有样本图像特征描述子保存在列表中

    des_list = []

    im = cv2.imread(image_path)

    im = cv2.resize(im, (300, 300))

    kpts = sift.detect(im)

    kpts, des = sift.compute(im, kpts)

    des_list.append((image_path, des))

    descriptors = des_list[0][1]

    for image_path, descriptor in des_list[0:]:

        descriptors = np.vstack((descriptors, descriptor))

    test_features = np.zeros((1, k), "float32")

    words, distance = vq(des_list[0][1], voc)

    for w in words:

        test_features[0][w] += 1



    # 归一化特征向量

    test_features = stdSlr.transform(test_features)

    # 进行预测，返回预测结果

    predictions = [classes_names[i] for i in clf.predict(test_features)]

    return predictions





if __name__ == "__main__":

    test_path = "../input/flowers/flowers/test"

    testing_names = os.listdir(test_path)

    image_paths = []

    class_path = []

    true_path = []

    n = 0

    for training_name in testing_names:

        dir = os.path.join(test_path, training_name)

        for i in os.listdir(dir):

            temp = os.path.join(dir, i)

            class_path.append(temp)

        image_paths += class_path

        true_path += [training_name] * len(class_path)

        class_path = []



    for i in range(len(image_paths)):

        predictions = predict_image(image_paths[i])

        print("真实类别为: {}, 预测类别为: {}".format(true_path[i], predictions[0]))

        if predictions[0] == true_path[i]:

            n += 1

    print("模型的准确率为{}".format(n / len(true_path)))





from IPython.display import FileLink

FileLink("bow.pkl")
