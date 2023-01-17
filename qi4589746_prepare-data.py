import numpy as np

import scipy.io as sio 

import os

import cv2

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K

import matplotlib.pyplot as plt

import pprint

#滿滿的lib，沒用到的就先別在意了  (´・ω・`)
annos = sio.loadmat('../input/devkit/devkit/cars_train_annos.mat')

pprint.pprint(annos) #mat使用python讀近來的格式
pprint.pprint(annos["annotations"][:,0]) #對應到作業中要求的資料格式，[bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname]

# 因為是超多層array，看起來會有些亂，下面的資料簡化後其實就等於： { "bbox_x1": 39, "bbox_y1": 116, "bbox_x2": 569, "bbox_y2": 375, "class": 14, "fname": "00001.jpg"}
for i in range(6):

    pprint.pprint(annos["annotations"][:,0][0][i][0]) # 將其中一列的5個欄位抓出，由上到下分別是bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname，注意有的還是個陣列格式....(╯°Д°)╯ ┻━┻
path = annos["annotations"][:,0][0][5][0].split(".")

pprint.pprint(int(path[0]) - 1)  # 用檔名000XX.jpg換成整數當作array index  (っ´ω`c)
def get_labels():

    annos = sio.loadmat('../input/devkit/devkit/cars_train_annos.mat')

    _, total_size = annos["annotations"].shape

    print("total sample size is ", total_size)

    labels = np.zeros((total_size, 5))

    for i in range(total_size):

        path = annos["annotations"][:,i][0][5][0].split(".")

        id = int(path[0]) - 1

        for j in range(5):

            labels[id, j] = int(annos["annotations"][:,i][0][j][0])

    return labels

labels = get_labels()

pprint.pprint(labels) # 將所有欄位簡化至一個array中 ξ( ✿＞◡❛)
print(labels[0]) #精簡資料其中一欄的數據 bbox_x1, bbox_y1, bbox_x2, bbox_y2, class。    fname勒？！？！(((ﾟДﾟ;)))     那傢伙已經變array的index了(最左邊那個0) ξ( ✿＞◡❛)
image_names = os.listdir("../input/cars_train/cars_train")

im = cv2.imread("../input/cars_train/cars_train/" + image_names[0])[:,:,::-1]

name = image_names[0].split('.')

image_label = labels[int(name[0]) - 1]

print("image is", image_names[0])

print("the label is " + str(image_label[4]))

plt.imshow(im)

plt.show()

x = im[int(image_label[1]):int(image_label[3]),int(image_label[0]):int(image_label[2])]

y = int(image_label[4])

print("努力裁剪中...ε≡ﾍ( ´∀`)ﾉ")

plt.imshow(x)

plt.show()