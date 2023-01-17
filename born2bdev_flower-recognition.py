# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow as tf
import h5py

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/flowers-recognition/flowers"))
input_dir = os.listdir("../input/flowers-recognition/flowers/flowers/sunflower/")
# print(input_dir)

# Any results you write to the current directory are saved as output.
sunflower = os.listdir('../input/flowers-recognition/flowers/flowers/sunflower/')
print(len(sunflower))
img_size = 139

x_train = np.zeros((734,img_size,img_size,3))
i=0

for sun in sunflower:
    img = cv2.imread('../input/flowers-recognition/flowers/flowers/sunflower/' + sun)
    x_train[i] = cv2.resize(img,(img_size,img_size))
    i += 1
print(x_train.shape)
labels_sunflower = np.zeros((734) )
for i in range(len(sunflower)):
    labels_sunflower[i] = 0
print(labels_sunflower.shape , labels_sunflower[1])

tulip = os.listdir('../input/flowers-recognition/flowers/flowers/tulip/')
print(len(tulip))

x_train_tulip = np.zeros((984,img_size,img_size,3))
i=0

for tu in tulip:
    img = cv2.imread('../input/flowers-recognition/flowers/flowers/tulip/' + tu)
    x_train_tulip[i] = cv2.resize(img,(img_size,img_size))
    i += 1
print(x_train_tulip.shape)

labels_tulip = np.zeros((984) )
for i in range(len(tulip)):
    labels_tulip[i] = 1
print(labels_tulip.shape , labels_tulip[1])
daisy = os.listdir('../input/flowers-recognition/flowers/flowers/daisy/')
print(len(daisy))
x_train_daisy = np.zeros((769,img_size,img_size,3))
i=0

for da in daisy:
    img = cv2.imread('../input/flowers-recognition/flowers/flowers/daisy/' + da)
    x_train_daisy[i] = cv2.resize(img,(img_size,img_size))
    i += 1
print(x_train_daisy.shape)

labels_daisy = np.zeros((769) )
for i in range(len(daisy)):
    labels_daisy[i] = 2
print(labels_daisy.shape , labels_daisy[1])
rose = os.listdir('../input/flowers-recognition/flowers/flowers/rose/')
print(len(rose))
x_train_rose = np.zeros((784,img_size,img_size,3))
i=0

for ro in rose:
    img = cv2.imread('../input/flowers-recognition/flowers/flowers/rose/' + ro)
    x_train_rose[i] = cv2.resize(img,(img_size,img_size))
    i += 1
print(x_train_rose.shape)

labels_rose = np.zeros((784) )
for i in range(len(rose)):
    labels_rose[i] = 3
print(labels_rose.shape , labels_rose[1])
dand = os.listdir('../input/flowers-recognition/flowers/flowers/dandelion/')
print(len(dand))
i=0
x=0

for da in dand:
    if 'jpg' in da:
        x += 1
x_train_dand = np.zeros((x,img_size,img_size,3))
print(x)

for da in dand:
    if 'jpg' in da:
        img = cv2.imread('../input/flowers-recognition/flowers/flowers/dandelion/' + da)
        x_train_dand[i] = cv2.resize(img,(img_size , img_size))
        i += 1
print(x_train_dand.shape)

labels_dand = np.zeros((x) )
for i in range(x):
    labels_dand[i] = 4
print(labels_dand.shape , labels_dand[1])
x_train = np.concatenate((x_train , x_train_tulip , x_train_daisy , x_train_rose , x_train_dand))
print(x_train.shape)
del x_train_tulip
del x_train_daisy
del x_train_rose
del x_train_dand
y_train = np.concatenate((labels_sunflower , labels_tulip , labels_daisy , labels_rose , labels_dand))
print(y_train.shape)
del labels_sunflower
del labels_tulip
del labels_daisy
del labels_rose
del labels_dand
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_train , y_train , test_size=0.1)
print(x_train.shape , x_test.shape)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten , Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.optimizers import adam
y_train = np_utils.to_categorical(y_train , 5)
y_test = np_utils.to_categorical(y_test , 5)
print(y_train[0] , y_test[0])
input_shape=(128,128,3)
num_classes = 5

model = Sequential()

model.add(Conv2D(32, (2,2) ,padding='same',input_shape=input_shape))
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Conv2D(256, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (2, 2)))
model.add(Conv2D(1024, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
model.compile(optimizer=adam(lr=2e-3),
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_test, y_test))
#model.save('model1.h5')
from keras.applications import InceptionResNetV2
conv_base = InceptionResNetV2(include_top=False , weights='../input/inception-resnet-v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5' , input_shape=(139,139,3))
x_train /= 255
x_test /= 255
x_train = conv_base.predict(x_train)
print(x_train[0])
x_test = conv_base.predict(x_test)
print(x_test[0])
from keras import models
from keras import optimizers

classes = 5
model = models.Sequential()
model.add(Flatten(batch_input_shape=(None , x_train.shape[1] , x_train.shape[2] , x_train.shape[3])))
model.add(Dense(classes))
model.add(Activation('softmax'))

model.summary()
model.compile(optimizer=optimizers.adam(lr=2e-3),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(x_test, y_test))
#model.save('../input/inception-resnet-v2/mymodel.h5')
