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
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from keras.preprocessing.image import save_img

import numpy as np

tf.compat.v1.enable_eager_execution()

def load_img(path_to_img):

    img = tf.io.read_file(path_to_img)

    img = tf.image.decode_jpeg(img, channels=3)

#   the following line will convert the image to float value ie between 0 and 1

    img = tf.image.convert_image_dtype(img, tf.float32)

#   the following line will resize the image to 256,256

    img = tf.image.resize(img, (256, 256))

#   the following line will add a new axis to the image

    img = img[tf.newaxis, :]

    return img



def tensor_to_image(tensor):

  tensor = tensor*255

  tensor = np.array(tensor.numpy(), dtype=np.uint8)

  if np.ndim(tensor)>3:

    assert tensor.shape[0] == 1

    tensor = tensor[0]

  return tensor



# from main.functions.functions import handle_uploaded_file

# Create your views here.

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3)))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(64, (3, 3), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(256, (3, 3), padding='same'))

# model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))#, beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

# model.add(Activation('relu'))

# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(layers.Conv2D(256, (3, 3), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))

model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,

									beta_initializer='zeros', gamma_initializer='ones',

									moving_mean_initializer='zeros',

									moving_variance_initializer='ones'))  # , beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(layers.Activation('relu'))

model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))

model.load_weights("https://res.cloudinary.com/dnn4bm8le/raw/upload/v1581172887/stylestore/model_62_tlbegc.h5")

model.load_wights("https://drive.google.com/file/d/1LoXY1BCT4NS0WdnvGwhhePvy3wl-zv5s/view?usp=sharing")
