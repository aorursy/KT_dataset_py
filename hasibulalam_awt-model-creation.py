# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

#from keras import optimizers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading data

import pickle

with open("../input/X.pickle", "rb") as fp:   # Unpickling

        X_feature = pickle.load(fp)

        

with open("../input/Y.pickle", "rb") as fp:   # Unpickling

        Y_label = pickle.load(fp)
#creating model 



X_feature= X_feature/255.0

model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=X_feature.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#

#model.add(Conv2D(64, (3, 3)))

#model.add(Activation('relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Activation('relu'))





model.add(Dense(4))

model.add(Activation('softmax'))

#model.add(Activation('sigmoid'))

#compileing model 

#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#adam_optimiser= optimizers.adam(lr=0.0001) 

#model.compile(adam_optimiser, loss='sparse_categorical_crossentropy',  metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

model.fit(x=X_feature, y=Y_label, batch_size=20, epochs=50, validation_split=0.1, shuffle=True)
model.save('image_classifier_002.model')
## Compile model

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## checkpoint

#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#callbacks_list = [checkpoint]

## Fit the model

#model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)

######https://machinelearningmastery.com/check-point-deep-learning-models-keras/
model.summary()
model.save('imagclass_001.h5')