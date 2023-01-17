import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
%tensorflow_version 2.x

from tensorflow import keras
import tensorflow as tf
tf.__version__
from keras.datasets import cifar10
# load dataset

(trainX, trainy), (testX, testy) = cifar10.load_data()
# summarize loaded dataset

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))

print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images

for i in range(9):

	# define subplot

	plt.subplot(331 + i)

	# plot raw pixel data

	plt.imshow(trainX[i])
trainX[10:20,10:20,10:20]
trainX[1,1:5,1:5,:]
# convert from integers to floats

#train_norm = trainX.astype('float32')

#test_norm = testX.astype('float32')

# normalize to range 0-1

train_norm = trainX / 255.0

test_norm = testX / 255.0
# one hot encode target values

from keras.utils import to_categorical

trainy = to_categorical(trainy)

testy = to_categorical(testy)
testy[1]
trainy[:5]
from keras import models

from keras.models import Sequential

from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
import keras
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 3)))  

model.add(keras.layers.MaxPooling2D((2, 2)))                                                            

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))                           

model.add(keras.layers.MaxPooling2D((2, 2)))                                                            

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))  

model.add(keras.layers.MaxPooling2D((2, 2)))     

model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))  

model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))  

model.add(keras.layers.Flatten())                                                                   

model.add(keras.layers.Dense(256, activation='relu',kernel_initializer="he_normal"))

model.add(Dropout(0.25))

model.add(keras.layers.Dense(128, activation='relu',kernel_initializer="he_normal")) 

model.add(Dropout(0.25))

model.add(keras.layers.Dense(10, activation='softmax')) 

                                                
model.summary()
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
# fit model

callback=keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3)

history=model.fit(train_norm, trainy, epochs=20, batch_size=512, validation_data=(test_norm, testy),callbacks=[callback])

# evaluate model

loss, acc = model.evaluate(test_norm, testy)

print('Test Loss : ', loss)

print('Test Accuracy:', acc)