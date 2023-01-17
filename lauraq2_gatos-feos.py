# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing import image

from zipfile import ZipFile

import sys

from tqdm import tqdm

import os





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Deep learning libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
directory = "../input/cats_and_dogs_small"



train_dir = directory + '/train'

test_dir = directory + '/validation'



train_dir_cats = directory + '/train/cats'

train_dir_dogs = directory + '/train/dogs'

test_dir_cats = directory + '/validation/cats'

test_dir_dogs = directory + '/validation/dogs'
print('number of cats training images - ',len(os.listdir(train_dir_cats)))

print('number of dogs training images - ',len(os.listdir(train_dir_dogs)))

print('number of cats testing images - ',len(os.listdir(test_dir_cats)))

print('number of dogs testing images - ',len(os.listdir(test_dir_dogs)))
data_generator = ImageDataGenerator(rescale = 1.0/255.0,

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    horizontal_flip=True,

                                    zoom_range=0.2)
batch_size = 64

training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   shuffle=True,

                                                   target_size = (400, 400),

                                                   batch_size = batch_size,

                                                   class_mode = 'binary')

#testing_data = data_generator.flow_from_directory(directory = test_dir,

#                                                  shuffle=True,

 #                                                 target_size = (400, 400),

  #                                                batch_size = batch_size,

   #                                               class_mode = 'binary')
import sys

from matplotlib import pyplot

from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers import Dense

from keras.layers import Flatten

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator#Se preparan las capas para la CNN

model = VGG16(include_top=False, input_shape=(400, 400, 3))

# mark loaded layers as not trainable

#Se marcan las capas cargadas como no entrenables

for layer in model.layers:

    layer.trainable = False

# Se añaden las nuevas capas del clasificador

flat1 = Flatten()(model.layers[-1].output)

class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

output = Dense(1, activation='sigmoid')(class1)

# define new model

model = Model(inputs=model.inputs, outputs=output)

# compile model

opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
fitted_model = model.fit_generator(training_data,

                        steps_per_epoch = len(training_data),

                        epochs = 20,

                        use_multiprocessing = False)
def testing_image(image_directory):

    test_image = image.load_img(image_directory, target_size = (400, 400))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(x = test_image)

    if result[0][0]  >= 1:

        #Dog

        prediction = 1

    else:

        #Cat

        prediction = 0

    return prediction
#print(testing_image(directory + '/test/'))



test = directory + '/test/'

lista = []

for images in os.listdir(test):

    lista.append(testing_image(test + images))

#Nos quedamos con los números para hacer el submission

import re

listaID = []

for item in os.listdir(test):

    listaID.append(re.sub("[^0-9]", "", str(item)))

    
def submission_generation(dataframe, name):

    """

    Esta función genera un csv a partir de un dataframe de pandas. 

    Con FileLink se genera un enlace desde el que poder descargar el fichero csv

    

    dataframe: DataFrame de pandas

    name: nombre del fichero csv

    """

    import os

    from IPython.display import FileLink

    os.chdir(r'/kaggle/working')

    dataframe.to_csv(name, index = False)

    return  FileLink(name)
print(len(lista))

print(len(listaID))



Predicting = pd.DataFrame({"id": listaID, 

                                      "label": lista})



submission_generation(Predicting , "TryPredict.csv")