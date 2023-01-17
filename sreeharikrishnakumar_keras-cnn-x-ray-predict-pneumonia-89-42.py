from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

import os
import numpy as np
import pandas as np

import matplotlib.pyplot as plt
%matplotlib inline
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.930):
      print("\nReached 93.0% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
print(os.listdir("../input/chest_xray/chest_xray"))

print(os.listdir("../input/chest_xray/chest_xray/train"))

print(os.listdir("../input/chest_xray/chest_xray/train/"))
# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = '../input/chest_xray/chest_xray/train'
validation_data_dir = '../input/chest_xray/chest_xray/val'
test_data_dir = '../input/chest_xray/chest_xray/test'

nb_train_samples = 5210
nb_validation_samples = 32
epochs = 20
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
##CNN_1
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
##CNN_2
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
##CNN_3
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
##CNN_4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
##CNN_5
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
##INPUT_LAYER
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.50))
##HIDDEN_LAYER_1
model.add(Dense(4))
model.add(Activation('relu'))
##OUTPUT_LAYER
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=128,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='binary')
model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=8,
    callbacks=[callbacks])
model.save_weights('first_try.h5')
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))