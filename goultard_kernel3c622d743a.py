import keras

import PIL as bodypillow

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import image

from matplotlib import pyplot

from PIL import Image

from os import listdir

from keras.preprocessing.image import ImageDataGenerator

from keras.engine import Input, Model

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape, concatenate, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Activation, Input, Lambda

from keras.callbacks import LearningRateScheduler

from keras import backend as k 

from keras import optimizers

from keras.callbacks import ModelCheckpoint 

from keras.optimizers import RMSprop

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder

import scipy

import os



# create generator

datagen = ImageDataGenerator(validation_split=0.3)

# prepare an iterators for each dataset

train_it = datagen.flow_from_directory('D:/Image/',

        target_size=(256, 256),

        batch_size=5,

        class_mode='categorical')

val_it = datagen.flow_from_directory('D:/Image/', class_mode='binary')

test_it = datagen.flow_from_directory('D:/Image/', class_mode='binary')



# confirm the iterator works

batchX, batchy = train_it.next()

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))



##################################################



data_list     = os.listdir('D:/Image/')

NUM_CLASSES   = len(data_list)

BATCH_SIZE    = 5

EPOCHS        = 5

CATEGORIES    = ['clear','cloudy','haze','partly_cloudy']



batch_size = 5

num_classes = 4

epochs = 4 # small number of epochs to reduce the computational time



# Training Parameters

learning_rate = 0.001



# Network Parameters

num_input = 784 # MNIST data input (img shape: 28*28)

num_classes = 4 # MNIST total classes (0-9 digits)

dropout = 0.25 # Dropout, probability to drop a unit



# input image dimensions

img_width, img_height = 256, 256



def lr_decay(epoch):

  return 0.01 * math.pow(0.666, epoch)

callback_learning_rate = LearningRateScheduler(lr_decay, verbose=True)



callback_is_nan = tf.keras.callbacks.TerminateOnNaN()



class PlotLosses(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        plt.legend()

        plt.show()

        

plot_losses = PlotLosses()



bnmomemtum=0.85

def fire(x, squeeze, expand):

  y  = Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)

  y  = BatchNormalization(momentum=bnmomemtum)(y)

  y1 = Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)

  y1 = BatchNormalization(momentum=bnmomemtum)(y1)

  y3 = Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)

  y3 = BatchNormalization(momentum=bnmomemtum)(y3)

  return concatenate([y1, y3])



def fire_module(squeeze, expand):

  return lambda x: fire(x, squeeze, expand)



x = Input(shape=[img_width, img_width, 3])

y = BatchNormalization(center=True, scale=False)(x)

y = Activation('relu')(y)

y = Conv2D(kernel_size=5, filters=12, padding='same', use_bias=True, activation='relu')(x)

y = BatchNormalization(momentum=bnmomemtum)(y)



y = fire_module(12, 24)(y)

y = MaxPooling2D(pool_size=2)(y)



y = fire_module(24, 48)(y)

y = MaxPooling2D(pool_size=2)(y)



y = fire_module(32, 64)(y)

y = MaxPooling2D(pool_size=2)(y)



y = fire_module(24, 48)(y)

y = MaxPooling2D(pool_size=2)(y)



y = fire_module(18, 36)(y)

y = MaxPooling2D(pool_size=2)(y)



y = fire_module(12, 24)(y)



y = GlobalAveragePooling2D()(y)

y = Dense(NUM_CLASSES, activation='sigmoid')(y)

model = Model(x, y)

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()



STEP_SIZE_TRAIN=train_it.n // train_it.batch_size 

STEP_SIZE_VALID=test_it.n // test_it.batch_size

history = model.fit_generator(

      train_it,

      steps_per_epoch=STEP_SIZE_TRAIN, 

      epochs=EPOCHS,

      validation_data=test_it,

      validation_steps=STEP_SIZE_VALID,

      callbacks=[plot_losses, callback_is_nan]) # callback_learning_rate,



accuracy = history.history['acc']

loss = history.history['loss']

val_accuracy = history.history['val_acc']

val_loss = history.history['val_loss']



print(f'Training Accuracy: {np.max(accuracy)}')

print(f'Training Loss: {np.min(loss)}')

print(f'Validation Accuracy: {np.max(val_accuracy)}')

print(f'Validation Loss: {np.min(val_loss)}')
import os

os.chdir("/working")