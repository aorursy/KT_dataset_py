from __future__ import absolute_import

from __future__ import print_function

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pandas as pd

import numpy as np

from keras.utils import np_utils 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras import optimizers

from keras import losses

from keras.models import load_model

from keras import regularizers

import time

from keras import initializers
#Load the training dataset ~87K states

all_train = pd.read_csv("../input/applied-ai-assignment-2/Assignment_2_train.csv")

all_train.loc[(all_train.state == 4),'state']=0

all_train.loc[(all_train.state == 5),'state']=1
len(all_train)
all_train[1:5]
#Create a train/validation split

data_to_use = 1

train=all_train[:int(len(all_train)*data_to_use)]

split = .9



Train = train[:int(len(train)*split)]

Valid = train[int(len(train)*split):]





#Remove the first and last column from the data, as it is the board name and the label

X_train = Train.iloc[:, 1:-1].values

X_valid = Valid.iloc[:, 1:-1].values



#Remove everything except the last column from the data, as it is the label and put it in y

y_train = Train.iloc[:, -1:].values

y_valid = Valid.iloc[:, -1:].values

len(X_train)
X_train[20].reshape(6,7)
print(X_train.shape)

print(X_valid.shape)
sample_train = X_train.reshape(-1,6,7)

X_train = sample_train.reshape(79062,6,7,1)

sample_valid = X_valid.reshape(-1,6,7)

X_valid = sample_valid.reshape(8785,6,7,1)



print(X_train.shape)

print(X_valid.shape)
#set input to the shape of one X value

dimof_input = X_train.shape[1]



# Set y categorical

dimof_output = int(np.max(y_train)+1)

y_train = np_utils.to_categorical(y_train, dimof_output)

y_valid = np_utils.to_categorical(y_valid, dimof_output)

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

#create model



model = Sequential()

model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(6,7,1)))

model.add(Flatten())



model.add(Dense(2, activation='softmax'))









model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
es = EarlyStopping(monitor='val_loss', #do not change

                                   mode='min',  #do not change

                                   verbose=1, # allows you to see more info per epoch

                                   patience=10) # **** patience is how many validations to wait with nothing learned (patience * validation_freq)



mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True) #do not change





history = model.fit(X_train, y_train, batch_size = 32, validation_data=(X_valid, y_valid),callbacks=[es, mc], epochs=1)