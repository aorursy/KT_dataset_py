# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2, l1
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_json('../input/planesnet.json')
data
x = []
for d in data['data']:
    d = np.array(d)
    x.append(d.reshape(( 3, 20 * 20)).T.reshape( (20,20,3) ))
x = np.array(x)
y = np.array(data['labels'])
print(x.shape)
print(y.shape)
# splitting the data into training ans test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

# Normalizing the data
scalar = MinMaxScaler()
scalar.fit(x_train.reshape(x_train.shape[0],-1))
x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# creating the convolutional model

def create_cnn(data_shape):

    kernel_size = 3
    
    model = Sequential()

    model.add(Conv2D(16, (kernel_size), strides=(1,1), padding='valid',input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(MaxPooling2D((2,2)))


    model.add(Conv2D(32, (kernel_size), strides=(1,1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (kernel_size), strides=(1,1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#     model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (kernel_size), strides=(1,1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (kernel_size), strides=(1,1), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model

model = create_cnn(x_train.shape)
print(model.summary())

# training the model
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#    if epoch:
#        lrate = initial_lrate/np.sqrt(epoch)
#    else:
#        return initial_lrate
    return lrate
opt = Adam(lr=0.0001)
lrate = LearningRateScheduler(step_decay)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=50, shuffle=True, verbose=2 ,
                               validation_data=(x_test, y_test), callbacks=[lrate],
                               )

# plotting the learning curves.
fig, ax = plt.subplots(1,2)
fig.set_size_inches((15,5))
ax[0].plot(range(1,51), history.history['loss'], c='blue', label='Training loss')
ax[0].plot(range(1,51), history.history['val_loss'], c='red', label='Validation loss')
ax[0].legend()
ax[0].set_xlabel('epochs')

ax[1].plot(range(1,51), history.history['acc'], c='blue', label='Training accuracy')
ax[1].plot(range(1,51), history.history['val_acc'], c='red', label='Validation accuracy')
ax[1].legend()
ax[1].set_xlabel('epochs')

