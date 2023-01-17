import cv2 # 机器视觉库，安装请用pip3 install opencv-python

import numpy as np # 数值计算库

import os # 系统库

from random import shuffle # 随机数据库 

from tqdm import tqdm # 输出进度库

import matplotlib.pyplot as plt # 常用画图库
train_dir = '../input/train/'

test_dir = '../input/test/'

img_size = 50

lr = 1e-3
train_data = create_train_data()
def process_test_data():

    testing_data = []

    for img in tqdm(os.listdir(test_dir)):

        if (not img.endswith('.jpg')):

            continue

        path = os.path.join(test_dir,img)

        img_num = img.split('.')[0]

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_size, img_size))

        testing_data.append([np.array(img), img_num])

        

    shuffle(testing_data)

    return testing_data
convnet = input_data(shape = [None, img_size, img_size, 1], name = 'input')
model = tflearn.DNN(convnet, tensorboard_dir='log')
train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)

y = [i[1] for i in train]

Xtest = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)

ytest = [i[1] for i in test]
test_data = process_test_data()