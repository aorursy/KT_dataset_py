#-*- coding: latin1 -*-

import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt



from keras.datasets import cifar10

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD

from keras.layers import Dropout

from keras.layers import BatchNormalization



plt.rcParams['figure.figsize'] = [15, 10]
# load train and test dataset

def load_dataset():

    # load dataset

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    

    # one hot encode target values (transform integer into a 10 element binary vector with a a1 for the class index of the value)

    trainY = np_utils.to_categorical(trainY)

    testY = np_utils.to_categorical(testY)

    return trainX, trainY, testX, testY





# scale pixels

def scale_pixels(train, test):

    # convert: integers -> float32

    train_norm = train.astype('float32')

    test_norm = test.astype('float32')

    

    # normalize to range 0-1

    train_norm = train_norm / 255.0

    test_norm = test_norm / 255.0

    

    # return normalized images

    return train_norm, test_norm





# plot diagnostic learning curves

def show_summary(history):

    # plot loss

    plt.subplot(211)

    plt.title('Cross Entropy Loss')

    plt.plot(history.history['loss'], color='blue', label='train')

    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy

    plt.subplot(212)

    plt.title('Classification Accuracy')

    plt.plot(history.history['acc'], color='blue', label='train')

    plt.plot(history.history['val_acc'], color='orange', label='test')



    

# run test

def run_test(mod, iterations = None):

    # load dataset

    trainX, trainY, testX, testY = load_dataset()

    

    # scale pixels

    trainX, testX = scale_pixels(trainX, testX)

    

    if iterations is None:

        iterations = 100

    

    # fit model

    history = mod.fit(trainX, trainY, 

                        epochs = iterations, 

                        batch_size = 64, 

                        validation_data = (testX, testY), 

                        verbose = 1)



    # evaluate model

    _, acc = mod.evaluate(testX, testY, verbose = 0)



    # print accuracy

    print('Accuracy (on testing set): > %.3f' % (acc * 100.0))

    

    # return history

    return history
# load the dataset

trainX, trainY, testX, testY = load_dataset()



# dataset summary

print('Training data: X = %s, y = %s' % (trainX.shape, trainY.shape))

print('Testing data: X = %s, y = %s' % (testX.shape, testY.shape))
# plot sample images

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # plot raw pixel data

    plt.imshow(trainX[i])

plt.show()
trainX, trainY = scale_pixels(trainX, trainY)
# define cnn model

def define_model_v1():

    # create sequential model

    model = Sequential()

    

    # add convolution

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # flatten (flattens input into a single vector)

    model.add(Flatten())

    

    # fully connected layer (128 units, ReLU activation function)

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    

    # fully connected layer (10 units, softmax activation function)

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = SGD(lr = 0.001, momentum = 0.9)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    return model



model = define_model_v1()



# run test

history = run_test(model)
# show summary

show_summary(history)
# define cnn model

def define_model_v2():

    # create sequential model

    model = Sequential()

    

    # add convolution (1st VGG block)

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add convolution (2st VGG block)

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # flatten (flattens input into a single vector)

    model.add(Flatten())

    

    # fully connected layer (128 units, ReLU activation function)

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    

    # fully connected layer (10 units, softmax activation function)

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = SGD(lr = 0.001, momentum = 0.9)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    return model



model = define_model_v2()



# run test

history = run_test(model)
# show summary

show_summary(history)
# define cnn model

def define_model_v3():

    # create sequential model

    model = Sequential()

    

    # add convolution (1st VGG block)

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add convolution (2st VGG block)

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add convolution (3st VGG block)

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # flatten (flattens input into a single vector)

    model.add(Flatten())

    

    # fully connected layer (128 units, ReLU activation function)

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    

    # fully connected layer (10 units, softmax activation function)

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = SGD(lr = 0.001, momentum = 0.9)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    return model



model = define_model_v3()



# run test

history = run_test(model)
# show summary

show_summary(history)
# define cnn model

def define_model_v3_dropout():

    # create sequential model

    model = Sequential()

    

    # add convolution (1st VGG block)

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add dropout

    model.add(Dropout(0.2))

    

    # add convolution (2st VGG block)

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add dropout

    model.add(Dropout(0.2))

    

    # add convolution (3st VGG block)

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    

    # add pooling

    model.add(MaxPooling2D((2, 2)))

    

    # add dropout

    model.add(Dropout(0.2))

    

    # flatten (flattens input into a single vector)

    model.add(Flatten())

    

    # fully connected layer (128 units, ReLU activation function)

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    

    # add dropout

    model.add(Dropout(0.2))

    

    # fully connected layer (10 units, softmax activation function)

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = SGD(lr = 0.001, momentum = 0.9)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    return model



model = define_model_v3_dropout()



# run test

history = run_test(model)
# show summary

show_summary(history)
# define cnn model

def define_model_v3_dropout_normalization():

    model = Sequential()

    

    # add convolution & batch normalization (1st VGG block)

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    

    # add pooling & dropout

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    

    # add convolution & batch normalization (2st VGG block)

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    

    # add pooling & dropout

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.3))

    

    # add convolution & batch normalization (3st VGG block)

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    

    # add pooling & dropout

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.4))

    

    # flatten (flattens input into a single vector)

    model.add(Flatten())

    

    # fully connected layer (10 units, softmax activation function)

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    # compile model

    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



model = define_model_v3_dropout_normalization()



# run test

history = run_test(model, 400)
# show summary

show_summary(history)