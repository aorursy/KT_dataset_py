import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import random
import glob
import cv2
import os

import math
import scipy
import tensorflow as tf 
from sklearn import metrics

from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# for Xception
from keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception

# for InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3

# for VGG-16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16

# for VGG-19
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19

# for ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

%matplotlib inline
train_data_path = '../input/10-monkey-species/training/training'
validation_data_path = '../input/10-monkey-species/validation/validation'
monkey_species = os.listdir(train_data_path)
img_width, img_height = 224, 224
batch_size = 4

print("Number of Categories:", len(monkey_species))
print("Categories: ", monkey_species)
train_datagen = ImageDataGenerator(
    rotation_range = 30,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
def history_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')

    plt.legend()
    plt.show()
Xception_base = Xception(weights='imagenet', include_top=False)

x = Xception_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(int(len(train_generator.class_indices.keys())), activation='softmax')(x)
Xception_transfer = models.Model(inputs=Xception_base.input, outputs=predictions)
Xception_transfer.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
history = Xception_transfer.fit_generator(train_generator,epochs=20,shuffle = True, verbose = 1, validation_data = validation_generator)
history_plot(history)
model_base = InceptionV3(weights='imagenet',include_top=False)

x = model_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512,activation='relu')(x)
predictions = layers.Dense(int(len(train_generator.class_indices.keys())) ,activation='softmax')(x)
InceptionV3_model = models.Model(inputs= model_base.input, outputs=predictions)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9) ,metrics=['accuracy'])
history = InceptionV3_model.fit_generator(train_generator, epochs=20, shuffle=True, verbose=1, validation_data=validation_generator)
history_plot(history)
model_base = VGG16(weights='imagenet',include_top=False)

x = model_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512,activation='relu')(x)
predictions = layers.Dense(int(len(train_generator.class_indices.keys())) ,activation='softmax')(x)
Vgg16_model = models.Model(inputs= model_base.input, outputs=predictions)
Vgg16_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9) ,metrics=['accuracy'])
history = Vgg16_model.fit_generator(train_generator, epochs=20, shuffle=True, verbose=1, validation_data=validation_generator)
history_plot(history)
model_base = VGG19(weights='imagenet',include_top=False)

x = model_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512,activation='relu')(x)
predictions = layers.Dense(int(len(train_generator.class_indices.keys())) ,activation='softmax')(x)
Vgg19_model = models.Model(inputs= model_base.input, outputs=predictions)
Vgg19_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9) ,metrics=['accuracy'])
history = Vgg19_model.fit_generator(train_generator, epochs=20, shuffle=True, verbose=1, validation_data=validation_generator)
history_plot(history)
model_base = ResNet50(weights='imagenet',include_top=False)

x = model_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512,activation='relu')(x)
predictions = layers.Dense(int(len(train_generator.class_indices.keys())) ,activation='softmax')(x)
Resnet50_model = models.Model(inputs= model_base.input, outputs=predictions)
Resnet50_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9) ,metrics=['accuracy'])
history = Resnet50_model.fit_generator(train_generator, epochs=20, shuffle=True, verbose=1, validation_data=validation_generator)
history_plot(history)