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
import tensorflow as tf

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set() # setting seaborn default for plots

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.datasets import mnist

# for Convolutional Neural Network (CNN) model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler

from keras import backend as K
K.set_image_dim_ordering('th')
train = pd.read_csv('../input/train.csv')
print (train.shape)
train.head()
test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()
# Separating the labels from training dataset and making it as x_label
y_train = train['label']
x_train = train.drop(labels=['label'],axis=1)
x_test = test

# Set values of the Data
x_train = x_train.values.astype('float32') # pixel values of all images in train set
y_train = y_train.values.astype('int32') # labels of all images
x_test = test.values.astype('float32') # pixel values of all images in test set
# fix random seed for reproducibility
random_seed = 7
np.random.seed(random_seed)
# one hot encode outputs'
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes
# Reshaping the Image for CNN 2-dimesional input in [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
num_pixels = x_train.shape[1]
print (num_pixels, x_train.shape, x_test.shape)
nn = 10
model = [0]*nn

for j in range(nn):
    model[j] = Sequential()
    model[j].add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(1, 28, 28), activation='relu',data_format='channels_first'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))
    
    # Compile model
    model[j].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# With data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nn
epochs = 50

# Fit the model
for j in range(nn):
    x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size = 0.10, random_state=random_seed)
    history[j] = model[j].fit_generator(datagen.flow(x_train2,y_train2, batch_size=64),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 0, steps_per_epoch=(len(x_train)//64),validation_steps=(len(x_val)//64),callbacks=[annealer])
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
results = np.zeros( (x_test.shape[0],10) )
for j in range(nn):
    results = results + model[j].predict(x_test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("ENSEMBLE.csv",index=False)
