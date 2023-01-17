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
import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential

from keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPool2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, concatenate

from keras.initializers import glorot_uniform, Constant

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.constraints import maxnorm

from keras.optimizers import SGD, Adam





X = np.load('../input/training-data/train_X.npy')

y = np.load('../input/training-data/train_y.npy')

test_X = np.load('../input/test-data/test_X.npy')

X = X/255

test_X = test_X/255

train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size = 0.3, random_state=21)
def model():

    images = Input((100, 100, 3))

    X = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(images)

    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Dropout(0.2)(X)

    X = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)

    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Dropout(0.2)(X)

    X = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)

    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Dropout(0.2)(X)

    X = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)

    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Dropout(0.3)(X)

    X = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer='glorot_uniform')(X)

    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Dropout(0.3)(X)

    X = Flatten()(X)

    X = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(X)

    X = Dropout(0.4)(X)

    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=images, outputs=X)

    # Compile model

    epochs = 25

    lrate = 0.0005

    adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())

    return model



model = model()



model.fit(train_X, train_y, validation_data=(dev_X, dev_y), epochs=100, batch_size=32)
print(model.history.history.keys())

import matplotlib.pyplot as plt

plt.figure(0)

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])

plt.legend(['accuracy', 'val_accuracy'])

plt.figure(1)

plt.plot(model.history.history['loss'])

plt.plot(model.history.history['val_loss'])

plt.legend(['loss', 'val_loss'])