# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



file_path = '/kaggle/input/catsanddogs/CATS_DOGS/'



# Any results you write to the current directory are saved as output.
import cv2

from keras.utils import np_utils

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import numpy as np



import warnings

warnings.filterwarnings('ignore')



import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.test.gpu_device_name()
# Descrição detalhada de cara argumento

# https://software.intel.com/en-us/articles/hands-on-ai-part-14-image-data-preprocessing-and-augmentation



from keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(

    rotation_range=30,      # Intervalo de rotação em graus

    width_shift_range=0.1,  # Para x < 1, pega um valor aleatório de uma distribuição normal no intervalo de uma fração da imagem e faz translação na altura

    height_shift_range=0.1, # Para x < 1, pega um valor aleatório de uma distribuição normal no intervalo de uma fração da imagem e faz translação na largura

    rescale=1/225,          # Multitplica os dados pelo valor colocado depois de fazer todas as transformações.

    shear_range=0.2,        # Controla taxa de deslocamento.

    zoom_range=0.2,         # Fator de zoom na imagem

    horizontal_flip=True,   # Flipa a imagem no eixo horizontal

    fill_mode='nearest'     # Atribui a cor do pixel mais próximo para o pixel que deveria estar em branco.

)
image_generator.flow_from_directory(file_path + 'train')
input_shape = (150, 150, 3)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128))

model.add(Activation('relu'))



model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
batch_size = 16



train_image_generator = image_generator.flow_from_directory(

    file_path + 'train',

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='binary'

)



test_image_generator = image_generator.flow_from_directory(

    file_path + 'test',

    target_size=input_shape[:2],

    batch_size=batch_size,

    class_mode='binary'

)
%%time



results = model.fit_generator(

    train_image_generator,

    epochs=100,

    steps_per_epoch=150,

    validation_data=test_image_generator,

    validation_steps=15

)
import seaborn as sns
fig, axs = plt.subplots(ncols=2, figsize=(16,8))



sns.lineplot(range(len(results.history['val_accuracy'])), results.history['val_accuracy'], ax=axs[0])

sns.lineplot(range(len(results.history['val_loss'])), results.history['val_loss'], ax=axs[1])

axs[0].set_xlabel('Epoch')

axs[1].set_xlabel('Epoch')

axs[0].set_ylabel('Accuracy')

axs[1].set_ylabel('Loss')



plt.show()
from keras.preprocessing import image



dog_file = file_path + 'train/DOG/80.jpg'

dog_img  = image.load_img(dog_file, target_size=(150,150))

dog_img  = image.img_to_array(dog_img)

dog_img  = np.expand_dims(dog_img, axis=0)

dog_img  = dog_img/255



predict_proba = model.predict(dog_img)

print(f'Probabilidade para cachorro: {predict_proba}')