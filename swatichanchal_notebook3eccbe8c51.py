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
from keras import layers

from keras import models

from keras import optimizers



import shutil

import os

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from keras.optimizers import Adam

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from keras.optimizers import Adam

from keras import regularizers
from tensorflow.keras.layers import  *
data_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')
data_df.head()
data_df.shape
test_dir ='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'

train_dir = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'



train_data = data_df[data_df['Dataset_type']=='TRAIN']

test_data = data_df[data_df['Dataset_type']=='TEST']
train_data.head()
test_data.head()
train_data.shape
test_data.shape
train_datagen = ImageDataGenerator(rescale=1./255,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(dataframe=train_data,

                                            directory=train_dir,

                                              x_col="X_ray_image_name",

                                            y_col="Label",

                                             target_size=(150, 150),

                                             batch_size=32,

                                             class_mode='binary',

                                              shuffle=True

                                             )

test_gen = test_datagen.flow_from_dataframe(dataframe=test_data,

                                            directory=test_dir,

                                            x_col="X_ray_image_name",

                                            y_col="Label",

                                             target_size=(150, 150),

                                             batch_size=32,

                                             class_mode='binary',

                                              shuffle=True

                                             )
model = models.Sequential()

model.add(base_model)

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'

                      ])
model.summary()
history = model.fit_generator(

    train_gen,

    steps_per_epoch=len(train_gen)/32,

    epochs=5,

    validation_data=test_gen,

    validation_steps=30)
base_model=tf.keras.applications.densenet.DenseNet121(include_top=False)
model = models.Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit_generator(

    train_gen,

    steps_per_epoch=len(train_gen)/32,

    epochs=15,

    validation_data=test_gen,

    validation_steps=30)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']
acc , loss , val_acc , val_loss
model.save_weights('./model_weights.h5')