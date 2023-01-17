# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



df = pd.read_csv('../input/aerial-cactus-identification/train.csv')



train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.15)



df['has_cactus'] = df['has_cactus'].astype(str)



train_generator = train_datagen.flow_from_dataframe(

        df,

        directory = '../input/aerial-cactus-identification/train/train',

        subset = 'training',

        x_col = 'id',

        y_col = 'has_cactus',

        target_size=(32, 32),

        batch_size=32,

        class_mode='binary')



validation_generator = train_datagen.flow_from_dataframe(

        df,

        directory = '../input/aerial-cactus-identification/train/train',

        subset = 'validation',

        x_col = 'id',

        y_col = 'has_cactus',

        target_size=(32,32),

        batch_size=32,

        class_mode = 'binary')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), input_shape=(32, 32, 3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.SpatialDropout2D(0.15),

    tf.keras.layers.Conv2D(32, (3,3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(64, (3,3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation('relu'),



    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation('relu'),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
from tensorflow.keras.optimizers import Adam



model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['acc'])
history = model.fit_generator(

      train_generator,

      steps_per_epoch=100,  

      epochs=20,

      verbose=1,

      validation_data = validation_generator,

      validation_steps=8)
import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.show()



plt.plot(history.history['val_loss'])

plt.show()
