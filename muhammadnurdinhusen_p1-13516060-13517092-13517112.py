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
#Get data

base_directory = '/kaggle/input/if4074-praktikum-1-cnn/P1_dataset'

print(base_directory)



train_directory = os.path.join(base_directory, 'train')

print(train_directory)



test_directory = os.path.join(base_directory, 'test')

print(test_directory)



train_cloudy_data = os.path.join(train_directory, '1')

print(train_cloudy_data)

train_rain_data = os.path.join(train_directory, '2')

print(train_rain_data)

train_shine_data = os.path.join(train_directory, '3')

print(train_shine_data)

train_sunrise_data = os.path.join(train_directory, '4')

print(train_sunrise_data)



total_trained = len(os.listdir(train_cloudy_data)) + len(os.listdir(train_rain_data)) + len(os.listdir(train_shine_data)) + len(os.listdir(train_sunrise_data))

print("total train data : " + str(total_trained))



total_test = len(os.listdir(test_directory))

print("total test data : " + str(total_test))
# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import preprocessing

from tensorflow.keras.preprocessing import image_dataset_from_directory



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

train_ds = image_dataset_from_directory(

directory=train_directory,

labels='inferred',

label_mode='categorical',

color_mode= 'rgb',

batch_size=64,

image_size=(256, 256))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([Conv2D(64, 3, input_shape=(256, 256, 3)),

                    Conv2D(64, 3, input_shape=(256, 256, 64)),

                    MaxPooling2D(pool_size=(112, 112), strides=(2,2), padding='valid'),

                    Flatten(),

                    Dense(512),

                    Dense(512),

                    Dense(4, activation='softmax'),

])
model.compile(

  'adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)
model.fit(

  train_ds, epochs=10

)
data = []

for directory, _, filenames in os.walk('/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/test'):

    for filename in filenames:

        img_path = directory + '/' + filename

        img = keras.preprocessing.image.load_img(

            img_path, target_size=(256, 256)

        )

        img_array = keras.preprocessing.image.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0) # Create a batch



        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        row = [filename, np.argmax(score)]

        data.append(row)

    print(data)
df = pd.DataFrame(data, columns=['id', 'label'])

df
df.to_csv ('P1_13516060_13517092_13517112.csv', index = False, header=True)
from tensorflow.keras.layers import Conv2D, MaxPooling3D, Dense, Flatten

model2 = Sequential([Conv2D(64, 3, input_shape=(256, 256, 3)),

                    MaxPooling2D(pool_size=(112, 112), strides=(2,2), padding='valid'),

                    Flatten(),

                    Dense(512),

                    Dense(512),

                    Dense(4, activation='softmax'),

])

model2.compile(

  'adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)

model2.fit(

  train_ds, epochs=10

)
data2 = []

for directory, _, filenames in os.walk('/kaggle/input/if4074-praktikum-1-cnn/P1_dataset/test'):

    for filename in filenames:

        img_path = directory + '/' + filename

        img = keras.preprocessing.image.load_img(

            img_path, target_size=(256, 256)

        )

        img_array = keras.preprocessing.image.img_to_array(img)

        img_array = tf.expand_dims(img_array, 0) # Create a batch



        predictions = model2.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        row = [filename, np.argmax(score)]

        data2.append(row)

    print(data2)
df2 = pd.DataFrame(data2, columns=['id', 'label'])

df.to_csv ('P1_13516060_13517092_13517112_2.csv', index = False, header=True)

df2