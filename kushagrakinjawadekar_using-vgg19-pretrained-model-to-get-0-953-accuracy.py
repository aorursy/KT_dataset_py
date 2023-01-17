# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.applications import VGG19

# importing the dependencies

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import models

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
conv_base = VGG19(weights='imagenet',

                 include_top=False,

                 input_shape=(150,150,3))
conv_base.summary()
model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation = 'relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()



print("This is no. of trainable weights before freezing the conv base :", len(model.trainable_weights))

conv_base.trainable = False

print("The no of trainable weights after freezing conv Base :", len(model.trainable_weights))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers
import os, shutil

# our dataset (only the train folder)



original_dataset_dir = "../input/dogs-vs-cats/train/train"    # we were asked to work with train part only for practice.



print('total images in train folder: ', len(os.listdir(original_dataset_dir)))

# Create a Directory where we’ll store our dataset

base_dir = "../dog-cat-small"

os.mkdir("../dog-cat-small")



# directories for the training, validation and test splits

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



# directory with training cat pictures

train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)



# directory with training dog pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



# directory with validation cat pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)



# directory with validation dog pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



# directory with test cat pictures

test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)



# directory with test dog pictures

test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)



# copies the first 8750 cat images to train_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(8750)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dst)



# copies the next 2500 cat images to validation_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(8750, 11250)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dst)



# copies the next 1250 cat images to test_cats_dir

fnames = ['cat.{}.jpg'.format(i) for i in range(11250, 12500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dst)



# copies the first 8750 dog images to train_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(8750)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dst)



# copies the first 2500 dog images to validation_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(8750, 11250)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dst)



# copies the first 1250 dog images to test_dogs_dir

fnames = ['dog.{}.jpg'.format(i) for i in range(11250, 12500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)
# Seeing the content count of the splits

print('total training cat images:', len(os.listdir(train_cats_dir)))

print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))

print('total test dog images:', len(os.listdir(test_dogs_dir)))
train_datagen = ImageDataGenerator(

                rescale = 1./255,

                rotation_range = 40,

                width_shift_range = 0.2,

                height_shift_range = 0.2,

                shear_range = 0.2,

                zoom_range = 0.2,

                horizontal_flip = True,

                fill_mode = 'nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

                  train_dir,

                  target_size  = (150,150),

                  batch_size = 20,

                  class_mode = 'binary')  



valid_generator = test_datagen.flow_from_directory(

                  validation_dir,

                  target_size  = (150,150),

                  batch_size = 20,

                  class_mode = 'binary')  

model.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=2e-5),metrics=['acc'])
batch_size= 20
history = model.fit_generator(

          train_generator,

          validation_data= valid_generator,

          validation_steps= 50,

          epochs = 10)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1,11)

plt.plot(epochs,acc,'bo',label = 'Training Acc')

plt.plot(epochs,val_acc,'b',label = 'Validation Acc')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label = 'Training Loss')

plt.plot(epochs,val_loss,'b',label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.figure()

#Unfreezing last 3 Convnet because they contain minutes features which are important:



conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

    if layer.name =='block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['acc'])

history = model.fit_generator(

          train_generator,

          steps_per_epoch=100,

          epochs=100,

          validation_data=valid_generator,

          validation_steps=50)
epochs = range(1,101)

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']
def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points

plt.plot(epochs,smooth_curve(acc), 'bo', label='Smoothed training acc')

plt.plot(epochs,smooth_curve(val_acc), 'b', label='Smoothed validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs,

smooth_curve(loss), 'bo', label='Smoothed training loss')

plt.plot(epochs,

smooth_curve(val_loss), 'b', label='Smoothed validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()



test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=20,class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print('test acc: ', test_acc)