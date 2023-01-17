from __future__ import print_function

import numpy as np 

import pandas as pd

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation, Dropout

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam ,RMSprop

from keras import backend as K

import matplotlib.pyplot as plt 

%matplotlib inline
batch_size = 128 # Model parametrelerinin her güncellemesi için kullanacağımız veri sayısını gösterir.

dropout = 0.45

epochs = 20

hidden_units = 256
(x_train, y_train), (x_test, y_test) = mnist.load_data()



print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

# Örnek 25 mnist digitlerini train datasetten çektik.

indexes = np.random.randint(0, x_train.shape[0], size=25)

images = x_train[indexes]

labels = y_train[indexes]



# Plot edelim.

plt.figure(figsize=(5,5))

for i in range(len(indexes)):

    plt.subplot(5, 5, i + 1)

    image = images[i]

    plt.imshow(image, cmap='gray')

    plt.axis('off')

    

plt.show()
# Label sayısını yani aslında kaç tane class'ımız olduğunu öğrenelim

num_labels = len(np.unique(y_train))

num_labels 
# One-Hot Encoding -> Class vektörlerini binary class matrislerine dönüştürelim.

y_train = keras.utils.to_categorical(y_train, num_labels)

y_test = keras.utils.to_categorical(y_test, num_labels)
# Verilerin daha iyi işlenmesi için normalize edelim.

image_size = x_train.shape[1]

input_size = image_size * image_size

input_size # boyutunu öğrendik



x_train = np.reshape(x_train, [-1, input_size])

x_train = x_train.astype('float32') / 255

x_test = np.reshape(x_test, [-1, input_size])

x_test = x_test.astype('float32') / 255
model = Sequential()

model.add(Dense(hidden_units, input_dim=input_size))

model.add(Activation('relu'))

model.add(Dropout(dropout))

model.add(Dense(hidden_units))

model.add(Activation('relu'))

model.add(Dropout(dropout))

model.add(Dense(num_labels))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)



loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print("\nTest accuracy: %.1f%%" % (100.0 * acc)) 