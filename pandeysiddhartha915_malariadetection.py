# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random

from shutil import copyfile

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_url = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/"



print(len(os.listdir(base_url + "Uninfected")))

print(len(os.listdir(base_url + "Parasitized")))
try:

    os.mkdir("/kaggle/images/")

    os.mkdir("/kaggle/images/train/")

    os.mkdir("/kaggle/images/val/")

    os.mkdir("/kaggle/images/train/Infected/")

    os.mkdir("/kaggle/images/val//Infected/")

    os.mkdir("/kaggle/images/train/Uninfected/")

    os.mkdir("/kaggle/images/val/Uninfected/")

except OSError as e:

    print(e)
TRAIN_DIR = "/kaggle/images/train/"

VAL_DIR = "/kaggle/images/val/"



TRAIN_INFECTED_DIR = "/kaggle/images/train/Infected/"

TRAIN_UNINFECTED_DIR = "/kaggle/images/train/Uninfected/"

VAL_INFECTED_DIR = "/kaggle/images/val/Infected/"

VAL_UNINFECTED_DIR = "/kaggle/images/val/Uninfected/"
train_split = 0.9

size = len(os.listdir(base_url + "Uninfected"))
train_size = int(size*train_split)



print(train_size)

print(size - train_size)
list_Uninfected = random.sample( os.listdir(base_url + "Uninfected"), size)

list_Infected = random.sample( os.listdir(base_url + "Parasitized"), size)



print(len(list_Uninfected))

print(len(list_Infected))
for  i in range(size):

    if i < train_size:

        copyfile(base_url + "Uninfected/" + list_Uninfected[i], TRAIN_UNINFECTED_DIR + list_Uninfected[i])

        copyfile(base_url + "Parasitized/" + list_Infected[i], TRAIN_INFECTED_DIR + list_Infected[i])

    else:

        copyfile(base_url + "Uninfected/" + list_Uninfected[i], VAL_UNINFECTED_DIR + list_Uninfected[i])

        copyfile(base_url + "Parasitized/" + list_Infected[i], VAL_INFECTED_DIR + list_Infected[i])
print(len(os.listdir(TRAIN_UNINFECTED_DIR)))

print(len(os.listdir(TRAIN_INFECTED_DIR)))

print(len(os.listdir(VAL_UNINFECTED_DIR)))

print(len(os.listdir(VAL_INFECTED_DIR)))
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   rotation_range = 40,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

                                   #fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size = (100, 100),

    #batch_size = 128,

    class_mode = 'binary')



validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(

    VAL_DIR,

    target_size = (100, 100),

    #batch_size = 16,

    class_mode = 'binary')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape = (100, 100, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (100, 100, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation="relu"),

    #tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="sigmoid")

])



model.compile(loss = "binary_crossentropy", optimizer='adam', metrics=['acc'])



model.summary()
history = model.fit_generator(train_generator,

                              epochs=15,

                              verbose=1,

                              validation_data=validation_generator)
import matplotlib.pyplot as plt





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