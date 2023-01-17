# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.listdir('./')



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/fer.csv')

print (df.values.shape)
train = df[["emotion", "pixels"]][df["Usage"] == "Training"]



train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))

x_train = np.vstack(train['pixels'].values)

y_train = np.array(train["emotion"])

x_train.shape, y_train.shape

public_test_df = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]



public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))

x_test = np.vstack(public_test_df["pixels"].values)

y_test = np.array(public_test_df["emotion"])
import tarfile

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization

from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
x_train = x_train.reshape(-1, 48, 48, 1).astype('float32')

x_test = x_test.reshape(-1, 48, 48, 1).astype('float32')



x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)



print(x_train.shape)

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (48 ,48,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(BatchNormalization())





model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (3 ,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(50, activation = "relu"))

model.add(Dropout(0.1))

model.add(Dense(7, activation = "softmax"))

model.summary()
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics=["accuracy"])



learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)



epochs = 3

batch_size = 128
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator( 

        rotation_range = 10, 

        zoom_range = 0.2,

        width_shift_range = 0.1, 

        height_shift_range = 0.1)



datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train, 

                                 y_train, 

                                 batch_size = batch_size),

                    epochs = 150, 

                    validation_data = (x_test,y_test),

                    verbose = 1, 

                    steps_per_epoch = x_train.shape[0] // batch_size,

                    callbacks = [learning_rate_reduction])
model.save('100.h5')