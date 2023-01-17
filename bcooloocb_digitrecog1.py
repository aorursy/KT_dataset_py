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
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:07:24 2018

@author: ADMIN
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#load training and testing csv files 
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#import required modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#input X: every columns but the first
df_train_x = df_train.iloc[:,1:] 
#output Y: only the first column
df_train_y = df_train.iloc[:,:1] 

#display images and labels of first 5 records
ax = plt.subplots(1,5)
for i in range(0,5):   #validate the first 5 records
    ax[1][i].imshow(df_train_x.values[i].reshape(28,28), cmap='gray')
    ax[1][i].set_title(df_train_y.values[i])

#build the model   
def cnn_model(result_class_size):
    model = Sequential()
    #use Conv2D to create our first convolutional layer, with 32 filters, 5x5 filter size, 
    #input_shape = input image with (height, width, channels), activate ReLU to turn negative to zero
    model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))
    #add a pooling layer for down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add another conv layer with 16 filters, 3x3 filter size, 
    model.add(Conv2D(16, (3, 3), activation='relu'))
    #set 20% of the layer's activation to zero, to void overfit
    model.add(Dropout(0.2))
    #convert a 2D matrix in a vector
    model.add(Flatten())
    #add fully-connected layers, and ReLU activation
    model.add(Dense(130, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #add a fully-connected layer with softmax function to squash values to 0...1 
    model.add(Dense(result_class_size, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model
  
#turn the label to 42000 binary class matrix 
arr_train_y = np_utils.to_categorical(df_train_y['label'].values)
#define the model output size and get the summary
model = cnn_model(arr_train_y.shape[1])
model.summary()  

#normalize 255 grey scale to values between 0 and 1 
df_test = df_test / 255
df_train_x = df_train_x / 255

#reshape training X and text x to (number, height, width, channels)
arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))
arr_test_x_28x28 = np.reshape(df_test.values, (df_test.values.shape[0], 28, 28, 1))

random_seed = 3
#validate size = 8%
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(arr_train_x_28x28, arr_train_y, test_size = 0.08, random_state=random_seed)

#define model callback
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                              factor=0.5,
                              patience=3, 
                              min_lr=0.00001)

#define image generator
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range 
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1  # randomly shift images vertically
        )

datagen.fit(split_train_x)

#train the model with callback and image generator
model.fit_generator(datagen.flow(split_train_x,split_train_y, batch_size=64),
                              epochs = 30, validation_data = (split_val_x,split_val_y),
                              verbose = 2, steps_per_epoch=700 
                              , callbacks=[reduce_lr])

#predict the result and save it as a csv file
prediction = model.predict_classes(arr_test_x_28x28, verbose=0)
data_to_submit = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)), "Label": prediction})
data_to_submit.to_csv("result.csv", header=True, index = False)

#validate the result by our own eyes
from random import randrange
#pick 10 images from testing data set
start_idx = randrange(df_test.shape[0]-10) 
  
fig, ax = plt.subplots(2,5, figsize=(15,8))
for j in range(0,2): 
  for i in range(0,5):
     ax[j][i].imshow(df_test.values[start_idx].reshape(28,28), cmap='gray')
     ax[j][i].set_title("Index:{} \nPrediction:{}".format(start_idx, prediction[start_idx]))
     start_idx +=1