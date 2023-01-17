# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))

# Set Keras image dim ordering
K.set_image_dim_ordering("th")
Train = pd.read_csv('../input/fashion-mnist_train.csv')
Test  = pd.read_csv('../input/fashion-mnist_test.csv')
print('========== Training Dataset ====================')
print(Train.describe())
print('========== Test Dataset ========================')
print(Test.describe())
IMG_ROWS = 28
IMG_COLS = 28
NB_CLASSES = 10
# Convert X matrix to float 
X_train = Train.iloc[:,1:].astype('float32').values / 255.0
X_test  = Test.iloc[:,1:].astype('float32').values / 255.0
print('==X_train==')
print(X_train[:5,:])
print('==X_test==')
print(X_test[:5,:])

# Convert Y array to categoricals
Y_train = np_utils.to_categorical(Train.iloc[:,0])
Y_test  = np_utils.to_categorical(Test.iloc[:,0])
print('==Y_train==')
print(Y_train[:5,:])
print('==Y_test==')
print(Y_test[:5,:])

# Number of rows
train_rows = X_train.shape[0]
test_rows  = X_test.shape[0]
print('train rows: ' + str(train_rows) +' test rows: ' + str(test_rows))

# In order to make the convolutional layer work, a new dimension must be added to the array so it represents 28x28 images
X_train = X_train[:, :, np.newaxis].reshape((train_rows, 28, 28))[:, np.newaxis, :, :]
X_test = X_test[:, :, np.newaxis].reshape((test_rows, 28, 28))[:, np.newaxis, :, :]

# Print array sizes
print('X_train shape: ' + str(X_train.shape) + ' X_test shape: ' + str(X_test.shape))
print('y_train shape: ' + str(Y_train.shape) + ' y_test shape: ' + str(Y_test.shape))
def create_dcnn_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25, seed=175))
    
    model.add(Conv2D(80, kernel_size=5, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25, seed=100))   
    
    model.add(Flatten())
    
    model.add(Dense(768))
    model.add(Activation("relu"))
    model.add(Dropout(0.5, seed=150))
    
    model.add(Dense(768))
    model.add(Activation("relu"))
    model.add(Dropout(0.5, seed=150))
    
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model
model = create_dcnn_model(input_shape=(1, 28, 28), num_classes=10)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, amsgrad=True), metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=64, epochs=15, verbose=1, validation_split=0.25)
score = model.evaluate(X_test, Y_test, verbose=1)
print("Test score:", score[0])
print('Test accuracy:', score[1])
