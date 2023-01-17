import numpy as np

import pandas as pd 

import cv2

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

folder='/kaggle/input/test-dataset/Fire-Detection'
data=[]

for files in os.listdir(folder):

    path=os.path.join(folder,files)

    for img in os.listdir(path):

        try:

            img1=cv2.imread(os.path.join(path,img))

            img1=cv2.resize(img1,(224,224))

            data.append([img1,files])

        except Exception as es:

            pass

    
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

X_train.reshape(-1,224,224,1)

X_test.reshape(-1,224,224,1)
model = Sequential()

model.add(Convolution2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Convolution2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))



model.add(Flatten())

model.add(Dense(4096,activation="relu"))

model.add(Dense(4096,activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=20,epochs=10,validation_data=(X_test,Y_test))