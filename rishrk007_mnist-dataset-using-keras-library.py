import tensorflow as tf

from tensorflow import keras

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mnist=keras.datasets.mnist #load mnist dataset 

(x_train,y_train),(x_test,y_test)=mnist.load_data() # splitting the training and testing datasets
x_train.shape
x_test.shape
import matplotlib.pyplot as plt
plt.imshow(x_train[50],cmap='gray')
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')

x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# The training  and testing dataset is structured as a 3-dimensional array of instance, 

#image width and image height. For a multi-layer perceptron model we must reduce the images down 

#into a vector of pixels. In this case the 28×28 sized images will be 784 pixel input values

x_train.shape
x_test.shape
x_train[100]
x_train=x_train/255.0  #The pixel values are gray scale between 0 and 255, so normalizing it b/w 0-1

x_test=x_test/255.0
from keras import models

from keras import layers

from keras import optimizers

from keras.layers import Dropout

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train) # one hot encoding for output variable because it is multiclass classification(0-9 class)
y_test=np_utils.to_categorical(y_test)# one hot encoding for output variable because it is multiclass classification(0-9 class)
x_train.shape
y_train.shape
model=models.Sequential()
model.add(layers.Dense(784,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(784,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=256,verbose=2)