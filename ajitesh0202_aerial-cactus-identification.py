# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras import layers,models

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential, Model 

from keras import optimizers

import os

import cv2

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

train.has_cactus=train.has_cactus.astype(str)
train.head()
# Data paths

train_path = '../input/train/train/'

test_path = '../input/test/test/'
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_path,x_col='id',

                                            y_col="has_cactus",class_mode="binary",batch_size=batch_size,target_size=(150,150))



validation_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_path,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
epochs=5

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()