# TensorFlow and tf.keras

import tensorflow as tf

from keras import models

from keras import layers

from keras import datasets

from keras import callbacks

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





# Helper libraries

import numpy as np

import seaborn as sns

import pandas as pd

import random

import matplotlib.pyplot as plt

import copy

import functools

import time





train_data_import = pd.read_csv('../input/fashion-mnist_train.csv')

test_data_import = pd.read_csv('../input/fashion-mnist_test.csv')
# create timer decorator

def func_timer(func):

    @functools.wraps(func)

    def wrapper_func_timer(*args, **kwargs):

        start_time = time.perf_counter()

        do_something = func(*args, **kwargs)

        end_time = time.perf_counter()

        print (f"Runtime for function {func.__name__!r}: {(end_time-start_time):.4f} seconds")

        return do_something

    return wrapper_func_timer



# def to plot accuracy and loss of model training

def plot_model_training(train_results):

    epochs = range(1, len(train_results['val_loss'])+1)

    plt.plot(epochs, train_results['acc'], 'bo', label = 'Training acc')

    plt.plot(epochs, train_results['val_acc'], 'b-', label = 'Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, train_results['loss'], 'ro', label = 'Training loss')

    plt.plot(epochs, train_results['val_loss'], 'r-', label = 'Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show
# function to build CNN

def build_CNN():    

    model = models.Sequential()

    model.add(layers.BatchNormalization(input_shape=(28,28,1)))

    model.add(layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(10, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model



def build_CNN_V2():

    model = models.Sequential()

    model.add(layers.BatchNormalization(input_shape=(28,28,1)))

    model.add(layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(10, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model



def build_CNN_V3():

    model = models.Sequential()

    model.add(layers.BatchNormalization(input_shape=(28,28,1)))

    model.add(layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model



#running CNN without data augmentation

@func_timer

def run_CNN(train_data, train_labels, val_data, val_labels, model, epochs=20):

    checkpointer = callbacks.ModelCheckpoint(filepath='Fashion MNIST CNN.hdf5', save_best_only=True, verbose=1)

    reduce = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

    history = model.fit(train_data, train_labels, batch_size = 32, callbacks=[checkpointer],

                        epochs=epochs, verbose=2, validation_data=(val_data, val_labels))

    return model, history



#running CNN with data augmentation

@func_timer

def run_CNN_aug(datagen, train_data, train_labels, val_data, val_labels, model, epochs=20):

    checkpointer = callbacks.ModelCheckpoint(filepath='Fashion MNIST CNN_aug.hdf5', save_best_only=True, verbose=1)

    reduce = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

    

    history = model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32),

                                  steps_per_epoch=train_data.shape[0]//20, epochs=epochs, callbacks=[checkpointer, reduce],

                                  validation_data=(val_data, val_labels), verbose=2)

    return model, history
# Separate images and labels from the training and test .csv

train_data = train_data_import.iloc[:,1:]

train_labels = train_data_import.loc[:,'label']

test_data = train_data_import.iloc[:,1:]

test_labels = train_data_import.loc[:,'label']



#reshape for CNN

train_data = train_data.values.reshape(train_data.shape[0],28,28,1) / 255.

test_data = test_data.values.reshape(test_data.shape[0],28,28,1) / 255.



# encoding labels

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)



# split training into partial_train & val

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1)
init_model = build_CNN()

model, history = run_CNN(train_data, train_labels, val_data, val_labels, init_model, epochs=20)

plot_model_training(history.history)
init_model = build_CNN_V2()

model, history = run_CNN(train_data, train_labels, val_data, val_labels, init_model, epochs=50)

plot_model_training(history.history)
init_model = build_CNN_V3()

model, history = run_CNN(train_data, train_labels, val_data, val_labels, init_model, epochs=50)

plot_model_training(history.history)
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

datagen.fit(train_data)



init_model = build_CNN()

model, history = run_CNN_aug(datagen, train_data, train_labels, val_data, val_labels, init_model, epochs=50)

plot_model_training(history.history)
def plot_image(i, predictions_array, true_label, img):

    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])



    plt.imshow(img, cmap=plt.cm.binary)



    predicted_label = np.argmax(predictions_array)

    true_label = np.argmax(true_label)

    if predicted_label == true_label:

        color = 'blue'

    else:

        color = 'red'



    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0, 1]) 

    predicted_label = np.argmax(predictions_array)

    true_label = np.argmax(true_label)

    thisplot[predicted_label].set_color('red')

    thisplot[true_label].set_color('blue')
#reshape for CNN prediction

test_images = test_data.reshape(10000,28,28,1)

predictions = model.predict(test_images)



#reshape for plotting

test_images = test_images.reshape(10000,28,28)





xnum_rows = 6

num_cols = 5

#startpoint = 20

num_images = num_rows*num_cols

plt.figure(figsize=(3*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_image(i, predictions, test_labels, test_images)

    plt.subplot(num_rows, 2*num_cols, 2*i+2)

    plot_value_array(i, predictions, test_labels)

plt.show()