# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Normal_ = os.path.join('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA')

Pneumonia_ = os.path.join('../input/chest-xray-pneumonia/chest_xray/val/NORMAL')



Normal = os.listdir(Normal_)

print(Normal[:3])



Pneumonia = os.listdir(Pneumonia_)

print(Pneumonia[:3])
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



# Index for iterating over images

pic_index = 0

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)



pic_index += 8

normal = [os.path.join(Normal_, fname) 

                for fname in Normal[pic_index-8:pic_index]]

pneumonia = [os.path.join(Pneumonia_, fname) 

                for fname in Pneumonia[pic_index-8:pic_index]]



for i, img_path in enumerate(normal+pneumonia):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)



plt.show()
import tensorflow as tf

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)



# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/train',  # This is the source directory for training images

        target_size=(300, 300),  # All images will be resized to 150x150

        batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')
model.fit_generator(

      train_generator,  

      epochs=15,

      verbose=1)
#model.save_weights()
test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(

        '../input/chest-xray-pneumonia/chest_xray/test',  # This is the source directory for training images

        target_size=(300, 300),  # All images will be resized to 150x150

        batch_size=128,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')
scores = model.evaluate_generator(test_generator)

print("Accuracy = ", scores[1])