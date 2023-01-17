# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
from keras.layers import Conv2D,MaxPool2D, Flatten, Dropout,Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K

# Any results you write to the current directory are saved as output.
Data = np.load("../input/mnist.npz")
x_train = Data['x_train']
x_test = Data['x_test']
y_train = Data['y_train']
y_test = Data['y_test']

x_train.shape
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float')
import matplotlib.pyplot as plt
%matplotlib inline
tempimg = x_test[2500].reshape(28,28)
plt.imshow(tempimg, cmap='gray')
x_train_n = x_train/255
x_test_n = x_test/255
num_of_classes = np.unique(y_train).shape[0]
num_of_classes
y_train_c = np_utils.to_categorical(y_train, num_of_classes)
y_test_c = np_utils.to_categorical(y_test, num_of_classes)
#Defining the model with convolution layer

#ConvolutionLayer:with 30 feature maps of size (5x5)
#Pooling layer taking the max over 2*2 patches.
#Convolutional layer with 15 feature maps of size 3×3.
#Pooling layer taking the max over 2*2 patches.
#Dropout layer with a probability of 20%.
#Flatten layer.
#Fully connected layer with 128 neurons and rectifier activation.
#Fully connected layer with 50 neurons and rectifier activation.
#Output layer


def largermodel():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape = (28,28,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model
model = largermodel()
model.fit(x_train_n,y_train_c, epochs=5, batch_size=100)