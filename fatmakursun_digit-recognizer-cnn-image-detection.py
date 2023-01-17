# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')                    

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

from collections import deque

import numpy as np
y_train = train['label']

x_train = train.drop(labels = ['label'], axis=1)
x_train.shape

image_index = 7777

print(y_train[image_index])
# Reshaping the array to 4-dims so that it can work with the Keras API

x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)
print(x_train.shape)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print('Number of images in x_train', x_train.shape[0])

print('Number of images in x_test', x_test.shape[0])
# Some examples

print(plt.imshow(x_train[5][:,:,0]))
print(plt.imshow(x_train[450][:,:,0]))
print(plt.imshow(x_train[4215][:,:,0]))
# Importing the required Keras modules containing model and layers

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers

model = Sequential()

model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

model.add(Dense(100, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10, activation=tf.nn.softmax))
#compiling the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=150, verbose = 1)
pred = model.predict(x_test)

results = np.argmax(pred,axis = 1)

results = pd.Series(results,name = 'Label')
sub = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



sub.to_csv("cnn_best_model.csv",index=False)