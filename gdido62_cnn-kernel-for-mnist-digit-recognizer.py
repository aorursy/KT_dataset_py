# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,Conv2D
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

#
# load train data (60000 images 28x28x8bit + 60000 labels)
# each row contains the label is on the first column  and 
# the image on the remaining 784 columns
#
train = pd.read_csv('../input/mnist_train.csv')
print(train.shape)
train.head()
#
# load test data (10000 inages 28x28x8bit)
#
test = pd.read_csv('../input/mnist_test.csv')
print(test.shape)
test.head()
#
x_train,y_train = data_prep(train)
x_test,y_test = data_prep(test)
#
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#
# 
# draw the images 6, 7, 8 on the screen with the corresponding labels
#
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i,:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(np.argmax(y_train[i]));
#
# build the CNN model
#
cnn_model = Sequential()
cnn_model.add(Conv2D(20,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
cnn_model.add(Conv2D(20,kernel_size=(3,3),activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))
# 
# compile the model
#
cnn_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#
# fit the model: batch size = 100, epochs = 4, split = 20%
#
cnn_model.fit(x_train, y_train, 
              batch_size = 128,
              epochs=2,
              validation_split=0.2)

# evaluate the model on test data
cnn_model.evaluate(x_test, y_test)
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_test[i,:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(np.argmax(y_test[i]));