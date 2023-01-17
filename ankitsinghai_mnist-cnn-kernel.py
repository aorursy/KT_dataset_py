# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten,Activation, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import numpy as np
import keras
import keras.utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load the data in Data Frame
train_df = pd.read_csv('../input/train.csv')
train_df.head()

# Basic info on Data Frame
train_df.info()
# Initialise input data for CNN
input_data = np.zeros((train_df.shape[0],28,28))

# Getting Label from data Frame to np.array
Y_train = np.array(train_df['label'])
Y_train = keras.utils.to_categorical(Y_train, 10)
# method to convert an DataFrame row to matrix of n_rows
def conv_df_row_to_matrix(n_rows,arr):
    arr = arr / 255
    return arr.reshape(n_rows,-1)
# drop column label from data Frame 
train_np = train_df.drop('label',axis=1)
for i in range(train_df.shape[0]):
    input_data[i] = conv_df_row_to_matrix(28, np.array(train_np.iloc[i]))
    
X_train = input_data.reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],1)    
print(X_train.shape)
print(Y_train.shape)
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(32,(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2)))

model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

filepath_val_acc="model_cnn_acc.best.hdf5"
checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_val_acc]

history = model.fit(X_train,Y_train,batch_size=64,epochs=30,callbacks=callbacks_list)
