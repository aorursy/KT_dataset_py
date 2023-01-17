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
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=45,

        zoom_range=0.2,

        horizontal_flip=False,

        width_shift_range=0.1,  

        height_shift_range=0.1

)
train_generator = train_datagen.flow_from_directory(

        '/kaggle/input/rockpaperscissors/rps-cv-images',

        target_size=(150, 150),

        color_mode="grayscale",

        batch_size=32,

        class_mode="categorical",

        shuffle=True,

            )
from keras import models

from keras.models import Sequential

import tensorflow as tf
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (5,5), activation=tf.nn.relu,input_shape=(150, 150, 1)),

    tf.keras.layers.BatchNormalization(),



    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu,padding = 'Same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu,padding = 'Same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dense(3, activation = tf.nn.softmax)

])
model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])
model.fit(train_generator,epochs=10)
model.save_weights('adi.h5')
tf.__version__