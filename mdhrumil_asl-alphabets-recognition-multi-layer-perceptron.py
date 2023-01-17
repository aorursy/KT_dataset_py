# Importing the libraries.

import tensorflow as tf
import cv2
from glob import glob
from matplotlib import pyplot as plt
import random
import math
import os
import numpy as np
from numpy.random import seed
seed(100)
from tensorflow import set_random_seed
set_random_seed(101)

# A utility function to display sample images from the dataset.


def plotSample(character):
    print("Samples images for letter " + character)
    basePath = '../input/asl_alphabet_train/asl_alphabet_train/'
    imagePath = basePath + character + '/**'
    pathData = glob(imagePath)
    
    plt.figure(figsize=(16,16))
    images = random.sample(pathData, 3)
    plt.subplot(1,3,1)
    plt.imshow(cv2.imread(images[0]))
    plt.subplot(1,3,2)
    plt.imshow(cv2.imread(images[1]))
    plt.subplot(1,3,3)
    plt.imshow(cv2.imread(images[2]))
    plt.colorbar()
    plt.show()
    return
plotSample('H')
dataPath = "../input/asl_alphabet_train/asl_alphabet_train"
resizeTuple = (64, 64)
resizeDim = (64, 64, 3)
numLabels = 29
batchSize = 64

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=0.1)

train_generator = data_generator.flow_from_directory(dataPath, target_size=resizeTuple, batch_size=batchSize, shuffle=True, subset="training")
val_generator = data_generator.flow_from_directory(dataPath, target_size=resizeTuple, batch_size=batchSize, subset="validation")
# A utility function to plot the loss and accuracy learning curves.

def plotCurves(history):
    # Plotting history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training accuracy', 'validation accuracy'], loc='upper right')
    plt.show()


    # Plotting history for losses
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.show()
model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=resizeDim),
    tf.keras.layers.Dense(15, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.002)),
    tf.keras.layers.Dense(15, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.002)),
    tf.keras.layers.Dense(29, activation = "softmax")
])

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

history = model_1.fit_generator(train_generator, epochs=10, steps_per_epoch = 1224,validation_data=val_generator, validation_steps = 136, verbose = 1)

plotCurves(history)
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=resizeDim),
    tf.keras.layers.Dense(15, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(15, activation = "relu"),
    tf.keras.layers.Dense(29, activation = "softmax")
])

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

history = model_2.fit_generator(train_generator, epochs=10, steps_per_epoch = 1224,validation_data=val_generator, validation_steps = 136, verbose = 1)

plotCurves(history)
# A utility function to define a learning_rate decay based on epoch schedule.

def scheduler(epoch):
    if epoch < 5:
        return 0.00001
    else:
        return 0.00001 * math.exp(0.1 * (5 - epoch))
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=resizeDim),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(29, activation = "softmax")
])

model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model_3.fit_generator(train_generator, epochs=10, steps_per_epoch = 1224,validation_data=val_generator, validation_steps = 136, callbacks = [callback], verbose = 1)

plotCurves(history)
# A utility function to define a learning_rate decay based on epoch schedule.

def scheduler(epoch):
    if epoch < 25:
        return 0.00001
    else:
        return 0.00001 * math.exp(0.1 * (25 - epoch))
model_4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=resizeDim),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(29, activation = "softmax")
])

model_4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model_4.fit_generator(train_generator, epochs=30, steps_per_epoch = 1224,validation_data=val_generator, validation_steps = 136, callbacks = [callback], verbose = 1)


plotCurves(history)