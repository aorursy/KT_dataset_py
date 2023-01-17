# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import keras as k

data=pd.read_csv('../input/fer2013/fer2013.csv')

# Any results you write to the current directory are saved as output.
train_set = data[(data.Usage == 'Training')] 

val_set = data[(data.Usage == 'PublicTest')]

test_set = data[(data.Usage == 'PrivateTest')] 
y_train=train_set['emotion']
X_train = np.array(list(map(str.split, train_set.pixels)), np.float32)



X_val = np.array(list(map(str.split, val_set.pixels)), np.float32)

X_test = np.array(list(map(str.split, test_set.pixels)), np.float32) 

Y_val=val_set['emotion']

y_test=test_set['emotion']

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 

X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
from keras.utils import np_utils,to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

len(y_train)
y_train = to_categorical(y_train, 7)

Y_val = to_categorical(Y_val, 7)

cnn4 = Sequential()

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

cnn4.add(BatchNormalization())



cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

cnn4.add(BatchNormalization())

cnn4.add(MaxPooling2D(pool_size=(2, 2)))

cnn4.add(Dropout(0.25))



cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

cnn4.add(BatchNormalization())

cnn4.add(Dropout(0.25))



cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

cnn4.add(BatchNormalization())

cnn4.add(MaxPooling2D(pool_size=(2, 2)))

cnn4.add(Dropout(0.25))



cnn4.add(Flatten())



cnn4.add(Dense(128, activation='relu'))

cnn4.add(BatchNormalization())

cnn4.add(Dropout(0.5))



cnn4.add(Dense(128, activation='relu'))

cnn4.add(BatchNormalization())

cnn4.add(Dropout(0.5))



cnn4.add(Dense(7, activation='softmax'))
cnn4.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])


#checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

cnn4.fit(X_train,

         y_train,

         batch_size=120,

         epochs=30,

         validation_data=(X_val, Y_val)

         )
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.0,  

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=0.1, 

        vertical_flip=0.1)  



datagen.fit(X_train)
history = cnn4.fit_generator(datagen.flow(X_train, y_train, batch_size=64),

                    validation_data=(X_val, Y_val),

                    steps_per_epoch=train_set.shape[0] // 64,

                    epochs = 40)