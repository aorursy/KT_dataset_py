!git clone https://github.com/krzysztofspalinski/deep-learning-methods-project-2.git
!mv deep-learning-methods-project-2 src
from keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data() 

x_train = x_train / 255
x_test = x_test / 255
from keras.utils.np_utils import to_categorical   

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

from src.scripts.residuallayer import ResnetIdentityBlock
# image_generator = ImageDataGenerator(rotation_range=30,
#                                      width_shift_range=0.05,
#                                      height_shift_range=0.05,
#                                      zoom_range=0.1,
#                                      horizontal_flip=True,
#                                      brightness_range=(0.9, 1.1))

image_generator = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0)

image_generator.fit(x_train)
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE, kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (2, 2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(image_generator.flow(x_train, y_train, batch_size=128),
                    validation_data=(x_test, y_test),
                    epochs=50)
