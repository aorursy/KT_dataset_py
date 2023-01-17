# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mnist=keras.datasets.mnist #load the dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() 

# splitting the dataset into training and testing datasets
x_train
import matplotlib.pyplot as plt
plt.imshow(x_train[40],cmap='gray')
x_train.shape
num_pixels = x_train.shape[1] * x_train.shape[2] # 28 * 28 = 784

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')

x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
x_train.shape
x_test.shape
x_train=x_train/255.0 

x_test=x_test/255.0

#The pixel values are gray scale between 0 and 255, so normalizing it b/w 0-1
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
y_train[40]
num_classes = y_test.shape[1]

num_classes
model=Sequential()
model.add(Conv2D(32,(5, 5), input_shape=(28, 28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
model.predict(x_test[:4])
y_test[:4]