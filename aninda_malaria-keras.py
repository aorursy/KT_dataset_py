# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BASE_DIR='/kaggle/input/cell_images/cell_images/'

PARASITIZED_DIR=os.path.join(BASE_DIR, 'Parasitized')

UNINFECTED_DIR=os.path.join(BASE_DIR, 'Uninfected')

print("No. of parasitized images: " + str(len(os.listdir(PARASITIZED_DIR))))

print("No. of uninfected images: " + str(len(os.listdir(UNINFECTED_DIR))))
try:

   os.makedirs('/tmp',exist_ok=True)

   os.makedirs('/tmp/malaria',exist_ok=True)

   os.makedirs('/tmp/malaria/training',exist_ok=True)

   os.makedirs('/tmp/malaria/training/parasitized',exist_ok=True)

   os.makedirs('/tmp/malaria/training/uninfected',exist_ok=True)

   os.makedirs('/tmp/malaria/testing',exist_ok=True)

   os.makedirs('/tmp/malaria/testing/parasitized',exist_ok=True)

   os.makedirs('/tmp/malaria/testing/uninfected',exist_ok=True)

except OSError:

    print("Error")

    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

  files = []

  for filename in os.listdir(SOURCE):

        file = SOURCE + filename

        if os.path.getsize(file) > 0:

            files.append(filename)

        else:

            print(filename + " is zero length, so ignoring.")



  training_length = int(len(files) * SPLIT_SIZE)

  testing_length = int(len(files) - training_length)

  shuffled_set = random.sample(files, len(files))

  training_set = shuffled_set[0:training_length]

  testing_set = shuffled_set[-testing_length:]



  for filename in training_set:

      this_file = SOURCE + filename

      destination = TRAINING + filename

      copyfile(this_file, destination)



  for filename in testing_set:

      this_file = SOURCE + filename

      destination = TESTING + filename

      copyfile(this_file, destination)





PARASITIZED_SOURCE_DIR = "/kaggle/input/cell_images/cell_images/Parasitized/"

TRAINING_PARASITIZED_DIR = "/tmp/malaria/training/parasitized/"

TESTING_PARASITIZED_DIR = "/tmp/malaria/testing/parasitized/"

UNINFECTED_SOURCE_DIR = "/kaggle/input/cell_images/cell_images/Uninfected/"

TRAINING_UNINFECTED_DIR = "/tmp/malaria/training/uninfected/"

TESTING_UNINFECTED_DIR = "/tmp/malaria/testing/uninfected/"



split_size = .9

split_data(PARASITIZED_SOURCE_DIR, TRAINING_PARASITIZED_DIR, TESTING_PARASITIZED_DIR, split_size)

split_data(UNINFECTED_SOURCE_DIR, TRAINING_UNINFECTED_DIR, TESTING_UNINFECTED_DIR, split_size)
print(len(os.listdir('/tmp/malaria/training/parasitized/')))

print(len(os.listdir('/tmp/malaria/testing/parasitized')))

print(len(os.listdir('/tmp/malaria/training/uninfected')))

print(len(os.listdir('/tmp/malaria/testing/uninfected')))

model = tf.keras.models.Sequential([

tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
TRAINING_DIR = "/tmp/malaria/training/"

train_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                    batch_size=100,

                                                    class_mode='binary',

                                                    target_size=(150, 150))



VALIDATION_DIR = "/tmp/malaria/testing/"

validation_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                              batch_size=100,

                                                              class_mode='binary',

                                                              target_size=(150, 150))
history = model.fit_generator(train_generator,

                              epochs=15,

                              verbose=1,

                              validation_data=validation_generator)
%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")





plt.title('Training and validation loss')