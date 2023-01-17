# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import struct as st

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from keras import layers

from keras.layers import Dropout , Input,GlobalAveragePooling2D, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.initializers import glorot_uniform

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import OneHotEncoder

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_types = {

        0x08: ('ubyte', 'B', 1),

        0x09: ('byte', 'b', 1),

        0x0B: ('>i2', 'h', 2),

        0x0C: ('>i4', 'i', 4),

        0x0D: ('>f4', 'f', 4),

        0x0E: ('>f8', 'd', 8)}
files = glob.glob("../input/*")

files
f = open(files[0],"rb")

f1 = open(files[1],"rb")

f2 = open(files[4] , "rb")

f3 = open(files[2] , "rb")
f.seek(0)

magic = st.unpack('>4B',f.read(4))

n = st.unpack('>I',f.read(4))[0] #num of images

nR = st.unpack('>I',f.read(4))[0] #num of rows

nC = st.unpack('>I',f.read(4))[0] #num of column

images = np.zeros((n,nR,nC))

nBytesTotal = n*nR*nC*1 #since each pixel data is 1 byte

images = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,f.read(nBytesTotal))).reshape((n,nR,nC))
plt.imshow(images[12])
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

X_train = np.zeros((60000,56,56))

for i in range(len(images)):

    img = images[i].astype("float32")

    img = cv2.resize(img,(56,56))

    img = cv2.filter2D(img , -1 , kernel)

    img = cv2.blur(img , (3,3))

    X_train[i] = img

   
plt.imshow(X_train[12])
f1.seek(8)

dataFormat = data_types[magic[2]][1]

dataSize = data_types[magic[2]][2]

train_label = np.asarray(st.unpack('>'+"B"*n,f1.read(n*1))).reshape((n,1))

enc = OneHotEncoder(sparse = False)

y_train = enc.fit_transform(train_label)
f2.seek(0)

magic = st.unpack('>4B',f2.read(4))

n = st.unpack('>I',f2.read(4))[0] #num of images

nR = st.unpack('>I',f2.read(4))[0] #num of rows

nC = st.unpack('>I',f2.read(4))[0] #num of column

images = np.zeros((n,nR,nC))

nBytesTotal = n*nR*nC*1 #since each pixel data is 1 byte

images_test = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,f2.read(nBytesTotal))).reshape((n,nR,nC))
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

X_test = np.zeros((10000,56,56))

for i in range(len(images_test)):

    img = images_test[i].astype("float32")

    img = cv2.resize(img,(56,56))

    img = cv2.filter2D(img , -1 , kernel)

    img = cv2.blur(img , (3,3))

    X_test[i] = img
plt.imshow(images_test[4])
plt.imshow(X_test[4])
X_test = np.reshape(X_test , (-1,56,56,1))
f3.seek(8)

dataFormat = data_types[magic[2]][1]

dataSize = data_types[magic[2]][2]

test_label = np.asarray(st.unpack('>'+"B"*n,f3.read(n*1))).reshape((n,1))

X_train = np.reshape(X_train,(60000,56,56,-1))
y_test = enc.transform(test_label)
inp = Input(shape = (56,56,1))

x = Conv2D(filters = 16 , kernel_size = (3,3),strides = (1,1),padding = "valid",kernel_initializer=glorot_uniform())(inp)

x = Activation("relu")(x)#54

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 32, kernel_size=(4,4) , strides = (2,2), padding="valid",kernel_initializer=glorot_uniform())(x)

x = Activation("relu")(x)#26

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(x)#13

x = Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding="valid", kernel_initializer=glorot_uniform())(x)

x = Activation("relu")(x)#10

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 128 , kernel_size = (4,4) , strides = (2,2), padding="valid",kernel_initializer=glorot_uniform())(x)

x = Activation("relu")(x)#4

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = GlobalAveragePooling2D()(x)

#x = Flatten()(x)

x = Dense(64)(x)

x = Activation("relu")(x)

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = Dense(32)(x)

x = Activation("relu")(x)

x = Dropout(0.2)(x)

x = Dense(10)(x)

out = Activation("softmax")(x)

mod = Model(inputs=inp,outputs=out)

mod.compile(loss = "categorical_crossentropy" , optimizer = "adam" , metrics=["accuracy"])

history = mod.fit(x=X_train,y=y_train,epochs=16,validation_split=0.15,shuffle=True)
mod.evaluate(X_test,y_test)
plt.plot(history.history['acc'])

plt.plot(history.history["val_acc"])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.xlabel("Loss")

plt.ylabel("Epochs")

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()