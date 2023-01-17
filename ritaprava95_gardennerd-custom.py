# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import numpy as np

import pandas as pd

import tensorflow as tf

import keras 

from keras.constraints import max_norm

from keras.utils import to_categorical

from keras.models import Model, Sequential

from keras.layers import Input, Conv2D, Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Activation, concatenate, GlobalAveragePooling2D

from keras.initializers import glorot_uniform

train_X = np.load("/kaggle/input/train_X.npy")



train_y = np.asarray(pd.read_csv("/kaggle/input/train.csv").iloc[:,1])



train_y = to_categorical(train_y)



image = Input(shape = [100, 100, 3])

X = Conv2D(128, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu')(image)

X = MaxPooling2D(pool_size=(2,2))(X)

X = Dropout(0.2)(X)

X = Conv2D(256, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu')(X)

X = MaxPooling2D(pool_size=(2,2))(X)

X = Dropout(0.3)(X)

X = Conv2D(256, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu')(X)

X = MaxPooling2D(pool_size=(2,2))(X)

X = Dropout(0.3)(X)

X = Conv2D(256, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu')(X)

X = MaxPooling2D(pool_size=(2,2))(X)

X = Dropout(0.3)(X)

X = Conv2D(512, kernel_size=(3,3), kernel_initializer='glorot_uniform', padding='same', activation='relu')(X)

X = MaxPooling2D(pool_size=(2,2))(X)

X = Dropout(0.4)(X)

X = Flatten()(X)

target = Dense(103, activation='softmax')(X)



model = Model(image, target, name='Sequential')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
model.fit(train_X, train_y, epochs=100, validation_split=0.2, batch_size=64)
import matplotlib.pyplot as plt 

print(model.history.history.keys())

plt.figure(1)

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])

plt.legend(['training accuracy', 'validation accuracy'])

plt.ylabel('Accuracy')

plt.xlabel('Epochs')
