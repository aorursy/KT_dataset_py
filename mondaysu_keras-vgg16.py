import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
from keras.layers import Input,Flatten,Dense,Dropout

from keras.models import Model

from keras.optimizers import SGD



# 28x28

from keras.datasets import mnist



#使用 OpenCV 进行图像处理

import cv2

import h5py as h5py

# state of the art 顶级模型

# 建立模型，使用vgg16 模型进行 fine-tuning(微调)，将顶层去掉，只保留其余的网络，include_top = False

# vgg16输入图片要求 48x48 --- 224x224

from keras.applications.vgg16 import VGG16



model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48,48,3))

for layer in model_vgg.layers:

    layer.trainable = False

# 获取到的是全连接之前的参数，需要flatten

x = Flatten(name='flatten')(model_vgg.output)

x = Dense(4096, activation='relu', name='fc1')(x)

x = Dense(4096, activation='relu', name='fc2')(x)

x = Dropout(0.5)(x)

y = Dense(10, activation='softmax')(x)



model_vgg_mnist = Model(inputs=model_vgg.input, outputs=y, name='vgg16')
# 打印模型结构

model_vgg_mnist.summary()

# 若改变输入图片大小，会影响全连接层的参数数量，但是不会影响卷积层的参数数量（权值共享）
# 设置优化器

sgd = SGD(lr=0.05, decay=1e-5)

# binary_crossentropy 二分类交叉熵   categorical_crossentropy 多分类交叉熵

model_vgg_mnist.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
(X_train, y_train),(X_test, y_test) = mnist.load_data("../test_data")
X_train, y_train = X_train[:10000], y_train[:10000]

X_test, y_test = X_test[:1000], y_test[:1000]

# 使用 OpenCV(use memory) resize图片，从3维的x_train中遍历出2维的i,然后将灰度图转变成rgb三通道图片

# interpolation小图片到大图片填充（插值方式）

# INTER_NEAREST 

# INTER_LINEAR 双线性插值

# INTER_AREA

# INTER_CUBIC

# INTER_LANCZOS4

X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB)

           for i in X_train]
import matplotlib.pyplot as plt

from PIL import Image



fig = plt.figure(figsize=(25, 16))

col = 0

for ii, img in enumerate(X_train):

    if col==32:

        break

    ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])

    plt.imshow(img)

    col += 1
# arr[np.newaxis] 给 arr添加一个维度，并且置为1

X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB)

          for i in X_test]

X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')



print(X_train.shape)

print(X_test.shape)



X_train = X_train / 255

X_test = X_test / 255





def tran_y(y):

    y_ohe = np.zeros(10)

    y_ohe[y] = 1

    return y_ohe
%%time

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])

y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])



model_vgg_mnist.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe),

                             epochs=100, batch_size=100)