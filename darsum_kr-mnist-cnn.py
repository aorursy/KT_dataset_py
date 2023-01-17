# coding = UTF-8

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn

import pandas as pd

import os

import sys

import time

from tensorflow import keras

from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing

from sklearn.preprocessing import StandardScaler

import h5py

for item in [tf,np,sklearn,pd,keras]:

    print(item.__name__,": ",item.__version__)
mnist = tf.keras.datasets.mnist

(x_train_all,y_train_all),(x_test,y_test) = mnist.load_data()

x_valid,x_train = x_train_all[:5000],x_train_all[5000:]

y_valid,y_train = y_train_all[:5000],y_train_all[5000:]



#x_valid = (x_valid.reshape(-1,784).astype('float32')/255.0).reshape(-1,28,28,1)

#x_train = (x_train.reshape(-1,784).astype('float32')/255.0).reshape(-1,28,28,1)

#x_test = (x_test.reshape(-1,784).astype('float32')/255.0).reshape(-1,28,28,1)





x_valid = x_valid.reshape(x_valid.shape[0],28,28,1).astype('float32')/255

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255

x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255



print(x_train.shape,y_train.shape)

print(x_valid.shape,y_valid.shape)

print(x_test.shape,y_test.shape)

model = keras.models.Sequential()

# 16,36,128

model.add(Conv2D(filters=32,

                 kernel_size=(5,5),

                 padding='same',

                 input_shape=(28,28,1), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,

                 kernel_size=(5,5),

                 padding='same',

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))




model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',metrics=['accuracy']) 

train_history=model.fit(x=x_train, 

                        y=y_train,validation_data=(x_valid,y_valid), 

                        epochs=25, batch_size=300,verbose=2)
model.evaluate(x_test,y_test)
from PIL import Image

path = "IMG/"

for file in os.listdir(path):

    #print(path+file)

    try:

        img = Image.open(path+file)

    except Exception as e:

        continue

    img_arr=np.array(img)

    img_arr = img_arr/255.0

    #print(img_arr.shape)

    pred=model.predict(img_arr.reshape(-1,28,28,1))

    

    print("瀹為檯鍊硷細",file[0],"- ","棰勬祴鍊硷細 ",pred.argmax()," ",file)