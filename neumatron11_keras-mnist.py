from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt



#adapted from  https://keras.io/examples/mnist_cnn/
# input image dimensions

img_rows, img_cols = 28, 28



num_classes = 10

# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()





x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



print("shape of each X array:")

print(x_train[1].shape)



print("targets")

print(y_train[0:10])



print("images")

plt.figure(figsize=(15,5))

for i in np.arange(10):

    plt.subplot(1,10,i+1)

    plt.imshow(x_train[i].reshape(28,28))

plt.show()

#scale X arrays from 0 to 255 down to 0 to 1

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# onehot encode targets

y_train_onehot = keras.utils.to_categorical(y_train, num_classes)

y_test_onehot = keras.utils.to_categorical(y_test, num_classes)



print("targets onehot")

print(y_train_onehot[0:5])
batch_size = 128

epochs = 12



""" 

This is the neural network

It is a relatively simple architecture - 2 convolutional layers to extract patterns in the images, a hidden dense layer (a single dense layer is essentially a logistic regression) and the final dense layer for predictions

the final dense layer represents the onehot encoding of each class

the model predicts the probability of the image being in each class



you train it in batches (how many images do you show it at a time) and epochs (how many times do you iterate through the whole training set)

"""

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train_onehot,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test_onehot))

score = model.evaluate(x_test, y_test_onehot, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
for i in np.arange(10):

    print("true Label: {}".format(y_test[i]))

    prediction_vector = model.predict(x_test[i].reshape(1,28,28,1))

    print("probability of each label {}".format(prediction_vector))

    label = np.argmax(prediction_vector)

    print("predicted label {}".format(label))

    print("image:")

    plt.imshow(x_test[i].reshape(28,28))

    plt.show()