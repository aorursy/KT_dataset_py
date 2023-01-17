import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import PIL
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
import random
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import save
from numpy import load

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten

training_path = "G:/Desktop/intel image classification/seg_train/seg_train/"
testing_path = "G:/Desktop/intel image classification/seg_test/seg_test/"
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                           zca_whitening=True,
                                                           rotation_range=40,
                                                           width_shift_range=0.15,
                                                           height_shift_range=0.15,
                                                           horizontal_flip=True
                                                           )
training_instances = generator.flow_from_directory(training_path, 
                                                   target_size=(150, 150),
                                                   batch_size=128)

generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_instances = generator.flow_from_directory(testing_path,
                                               target_size=(150, 150),
                                               batch_size=64)

IMG_SHAPE = (150, 150, 3)

base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
base_model.summary()
len(base_model.layers)

x = base_model.get_layer('block3_pool').output
x = tf.keras.layers.Flatten()(x)

x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
prediction = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
model.summary()
%%time
Batch_size=64
initial_epochs = 20
model_fit=model.fit(training_instances, 
                    steps_per_epoch=14034 // Batch_size,
                    epochs=initial_epochs,
                    validation_data=test_instances
                   )
model.evaluate(test_instances)

base_model.trainable = True
len(base_model.layers)
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 7

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
model.summary()
%%time
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(training_instances,
                         epochs=total_epochs,
                         initial_epoch =  model_fit.epoch[-1],
                         validation_data=test_instances)
model.evaluate(test_instances)

%%time
fine_tune_epochs1 = 10
total_epochs =  initial_epochs + fine_tune_epochs + fine_tune_epochs1

history_fine = model.fit(training_instances,
                         epochs=total_epochs,
                         initial_epoch =  history_fine.epoch[-1],
                         validation_data=test_instances)
model.evaluate(test_instances)

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 4

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
model.summary()
%%time
fine_tune_epochs2 = 5
total_epochs =  initial_epochs + fine_tune_epochs + fine_tune_epochs1 + fine_tune_epochs2

history_fine = model.fit(training_instances,
                         epochs=total_epochs,
                         initial_epoch =  history_fine.epoch[-1],
                         validation_data=test_instances)
model.evaluate(test_instances)

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 2

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"])
model.summary()
%%time
fine_tune_epochs3 = 10
total_epochs =  initial_epochs + fine_tune_epochs + fine_tune_epochs1 + fine_tune_epochs2 + fine_tune_epochs3

history_fine = model.fit(training_instances,
                         epochs=total_epochs,
                         initial_epoch =  history_fine.epoch[-1],
                         validation_data=test_instances)
model.evaluate(test_instances)
model.save('G:/Desktop/intel image classification/91%.h5')

