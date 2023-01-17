# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # open cv library
import matplotlib.pyplot as plt #library
# Importing necessary library
from keras.layers import Dense, Activation, Flatten,Dropout, Conv2D
from keras.models import Sequential
# import the necessary packages
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
trainingcsv_read=pd.read_csv('/kaggle/input/he-dl-trainimage/train.csv')

data= [] #np.empty((5983, 3, 60, 80), dtype='uint8')
labels= [] # np.empty((5983,), dtype='uint8')

import os
for dirname, _, filenames in os.walk('/kaggle/input/he-dl-trainimage/Train Images'):
    for filename in filenames:
        image = cv2.imread(os.path.join(dirname, filename))
        image = cv2.resize(image, (80,60))
        image = img_to_array(image)
        data.append(image)
        labels.append(trainingcsv_read['Class'][trainingcsv_read.loc[trainingcsv_read['Image']==filename].index[0]])

# Any results you write to the current directory are saved as output.



model=Sequential()
model.add(Conv2D(96,(3,3),padding='same',input_shape=(60,80,3),data_format="channels_last", activation='relu'))
# model.add(Activation('relu'))
# model.add(Conv2D(256,(3,3),padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(256,(3,3),padding='same',strides=(2,2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
model.summary()
from keras.optimizers import SGD
learning_rate=0.01
weight_decay=1e-6
momentum=0.9
model.compile(loss='binary_crossentropy',optimizer=SGD(lr=learning_rate,momentum=momentum,decay=weight_decay,nesterov=True),metrics=["accuracy"])
data = np.array(data) 
labels = np.array(labels)

from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

trainX = np.array(trainX)
trainY = np.array(trainY)
testX =  np.array(testX)
testY =  np.array(testY)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(len(trainX)//32)
# define additional training parameter
epochs =2
batch_size=32
# EPOCHS =2
# BS = 32
#fit the model
model.fit(trainX, trainY,epochs=epochs,batch_size=batch_size,validation_data=(testX, testY), verbose=1)
# model.fit_generator(
# 	aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY),
# 	steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)