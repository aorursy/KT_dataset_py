 

############ALEXNET

import numpy as np

import pandas as pd 

import cv2

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

folder='/kaggle/input/pothole-detection-dataset'
data=[]

i=0

for files in os.listdir(folder):

    path=os.path.join(folder,files)

    for img in os.listdir(path):

        try:

            img1=cv2.imread(os.path.join(path,img))

            img1=cv2.resize(img1,(227,227))

            data.append([img1,i])

        except Exception as es:

            pass

    i=i+1
data=np.array(data)

np.random.shuffle(data)

data_fea=[]

data_label=[]

for features,label in data:

    data_fea.append(features)

    data_label.append(label)
import gc

del data

gc.collect()
print(data_fea[0:10])
from keras.models import Sequential

from keras.layers import Flatten

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Dropout
data_fea=np.array(data_fea)

data_label=np.array(data_label)

print(data_fea.shape)

print(data_label.shape)

from sklearn.model_selection import train_test_split

X_train,X_test=train_test_split(data_fea,test_size=0.2)

Y_train,Y_test=train_test_split(data_label,test_size=0.2)

X_train.reshape(-1,227,227,1)

X_test.reshape(-1,227,227,1)
clf=Sequential()

clf.add(Convolution2D(96,(11,11),input_shape=(227,227,3),data_format='channels_last',strides=(3,3),activation='relu'))

clf.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

clf.add(Convolution2D(256,(5,5),strides=(1,1),padding='valid',activation='relu'))

clf.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

clf.add(Convolution2D(384,(3,3),strides=(1,1),padding='valid',activation='relu'))

clf.add(Convolution2D(384,(3,3),strides=(1,1),padding='valid',activation='relu'))

clf.add(Convolution2D(256,(3,3),strides=(1,1),padding='valid',activation='relu'))

clf.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

clf.add(Flatten())

clf.add(Dropout(0.5))

clf.add(Dense(4096,activation='relu'))

clf.add(Dropout(0.5))

clf.add(Dense(4096,activation='relu'))

clf.add(Dense(4096,activation='relu'))

clf.add(Dense(1,activation='sigmoid'))

clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

clf.fit(X_train,Y_train,batch_size=20,epochs=10,validation_data=(X_test,Y_test))