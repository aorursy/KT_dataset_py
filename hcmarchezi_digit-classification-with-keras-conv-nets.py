# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

import os
print(os.listdir("../input"))

K.set_image_dim_ordering("th")

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
Train, Test = train_test_split(df.values, test_size=0.2)
num_X_train = Train[:,1:].astype('float32') / 255.0
y_train = np_utils.to_categorical(Train[:,0])
print('-------------- X train ----------------')
print(num_X_train[:5,:])
print('-------------- y train ----------------')
print(y_train[:5,:])
num_X_test = Test[:,1:].astype('float32') / 255.0
y_test = np_utils.to_categorical(Test[:,0])
print('-------------- X test ----------------')
print(num_X_test[:5,:])
print('-------------- y test ----------------')
print(y_test[:5,:])
train_rows = num_X_train.shape[0]
test_rows  = num_X_test.shape[0]
print('train rows: ' + str(train_rows) +' test rows: ' + str(test_rows))

# In order to make the convolutional layer works a new dimension must be added to the array ( so it becomes 4)
X_train = num_X_train[:, :, np.newaxis].reshape((train_rows, 28, 28))[:, np.newaxis, :, :]
X_test = num_X_test[:, :, np.newaxis].reshape((test_rows, 28, 28))[:, np.newaxis, :, :]

print('X_train shape: ' + str(X_train.shape) + ' X_test shape: ' + str(X_test.shape))
print('y_train shape: ' + str(y_train.shape) + ' y_test shape: ' + str(y_train.shape))

#define the ConvNet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        
        # 1st stages - convolution followed by max-pooling
        
        # CONV => RELU => POOL
        # 20 convolutional filters, each one of which has a size of 5 x 5
        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second convolutional stage with ReLU activations followed  by a max-pooling.
        # 50 convolutional filters (should increase from previous layer) - common deep learning technique
        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # 2nd stages - dense or classifier layers
        
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return model with structure description
        return model    
# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10 # number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
# initialize the optimizer and model
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

