# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras

from PIL import Image

from sklearn import preprocessing

from keras.utils.np_utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class myCallback(keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('acc')>0.99):

            print('\n cancelling training!')

            self.model.stop_training=True
callbacks=myCallback()

data_train=pd.read_csv("../input/train.csv")

x_test=pd.read_csv("../input/test.csv")

x_train=data_train.iloc[:,1:]



#normalizing the training and testing data

x_train,x_test=x_train/255,x_test/255

y_train=data_train.iloc[:,0]



XTest1=x_test.values.reshape(x_test.shape[0],28,28)

#reshaping the training and testing data

XTrain = x_train.values.reshape(x_train.shape[0],28,28,1)

XTest = x_test.values.reshape(x_test.shape[0],28,28,1)



#converting to float type

#XTrain=XTrain.values.astype('float32')

#XTest=XTest.values.astype('float32')

#YTrain=y_train.values.astype('int32')



#relabeling as one hot

#YTrain=y_train

YTrain = to_categorical(y_train)
model=keras.Sequential([keras.layers.Conv2D(16,(3,3),input_shape=(28,28,1),activation='relu'),

                        keras.layers.MaxPooling2D(2,2),

                        keras.layers.Conv2D(16,(3,3),activation='relu'),

                        keras.layers.MaxPooling2D(2,2),

                        keras.layers.Flatten(),

                        keras.layers.Dense(128,activation='relu'),

                        keras.layers.Dense(10,activation='softmax')])

#using sparse_categorical_crossentropy as the labels are integers

#if labels are one hot encoded then use categorical_crossentropy

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(XTrain,YTrain,epochs=10,callbacks=[callbacks])
predictions_p=model.predict(XTest)

predictions_l=model.evaluate(XTrain,YTrain)
ind=199

print(predictions_p[ind])

print(predictions_l)
import matplotlib.pyplot as plt

plt.imshow(XTest1[ind],cmap=plt.get_cmap('gray'))

#plt.title(YTrain[0])