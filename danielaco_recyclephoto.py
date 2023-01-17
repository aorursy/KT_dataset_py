# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from fastai.vision import *
import sys

import os

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras import optimizers

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation

from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D

from tensorflow.python.keras import backend as K



K.clear_session()







data_entrenamiento = '../input/train/train'

data_validacion = '../input/test/test'



"""

Parameters

"""

epocas=20

longitud, altura = 100, 100

batch_size = 32

pasos = 1000

validation_steps = 300

filtrosConv1 = 32

filtrosConv2 = 64

tamano_filtro1 = (3, 3)

tamano_filtro2 = (2, 2)

tamano_pool = (2, 2)

clases = 2

lr = 0.0004





##Preparamos nuestras imagenes



entrenamiento_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.3,

    zoom_range=0.3,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)



entrenamiento_generador = entrenamiento_datagen.flow_from_directory(

    data_entrenamiento,

    target_size=(altura, longitud),

    batch_size=batch_size,

    class_mode='categorical')



validacion_generador = test_datagen.flow_from_directory(

    data_validacion,

    target_size=(altura, longitud),

    batch_size=batch_size,

    class_mode='categorical')



cnn = Sequential()

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamano_pool))



cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))

cnn.add(MaxPooling2D(pool_size=tamano_pool))



cnn.add(Flatten())

cnn.add(Dense(256, activation='relu'))

cnn.add(Dropout(0.5))

cnn.add(Dense(clases, activation='softmax'))



cnn.compile(loss='categorical_crossentropy',

            optimizer=optimizers.Adam(lr=lr),

            metrics=['accuracy'])









cnn.fit_generator(

    entrenamiento_generador,

    steps_per_epoch=pasos,

    epochs=epocas,

    validation_data=validacion_generador,

    validation_steps=validation_steps)

from keras.preprocessing.image import load_img, img_to_array

def predict(file):

  x = load_img(file, target_size=(longitud, altura))

  x = img_to_array(x)

  x = np.expand_dims(x, axis=0)

  array = cnn.predict(x)

  result = array[0]

  answer = np.argmax(result)

  if answer == 0:

    print("pred: carton")

  elif answer == 1:

    print("pred: plastico")

 



  return answer



img = open_image('../input/test/test/plastico/p170.jpg')

img
predict('../input/test/test/plastico/p170.jpg')
predict('../input/test/test/carton/c81.jpg')
img = open_image('../input/test/test/carton/c81.jpg')

img