import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import os

import cv2

import csv

import shutil

from glob import glob

from PIL import Image

from IPython.display import FileLink
print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/"))
!unzip -q ../input/dogs-vs-cats-redux-kernels-edition/train.zip
!ls train/ | wc -l
os.mkdir('valid/')
os.mkdir('train/cats/')

os.mkdir('train/dogs/')

os.mkdir('valid/cats/')

os.mkdir('valid/dogs/')
cats = glob('train/cat*.jpg')

shuf = np.random.permutation(cats)



for i in range(len(cats)): shutil.move(shuf[i], 'train/cats/')
!ls train/cats | wc -l
!ls train/dogs | wc -l
!ls train/ | wc -l
dogs = glob('train/dog*.jpg')

shuf = np.random.permutation(dogs)



for i in range(len(dogs)): shutil.move(shuf[i], 'train/dogs/')
!ls train/dogs | wc -l
!ls train/ | wc -l
cats_valid = glob('train/cats/*.jpg')

shuf = np.random.permutation(cats_valid)



for i in range(int(len(cats_valid) / 10)): shutil.move(shuf[i], 'valid/cats/')
dogs_valid = glob('train/dogs/*.jpg')

shuf = np.random.permutation(dogs_valid)



for i in range(int(len(dogs_valid) / 10)): shutil.move(shuf[i], 'valid/dogs/')
!ls train/cats | wc -l
!ls train/dogs | wc -l
import tensorflow as tf

import tensorflow.keras as keras
from keras.layers.normalization import BatchNormalization

from keras.utils.data_utils import get_file

from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD, RMSprop, Adam

from keras.preprocessing import image

from keras.models import model_from_json
datagen = image.ImageDataGenerator(horizontal_flip=True, zoom_range=0.5, shear_range=0.25, rotation_range=0.35, height_shift_range=0.25, 

                                   width_shift_range=0.25, brightness_range=(0.15, 0.30))

trn_batches = datagen.flow_from_directory('train/', target_size = (224, 224), class_mode = 'categorical', shuffle = True, batch_size = 64)



val_batches = datagen.flow_from_directory('valid/', target_size = (224, 224), class_mode = 'categorical', shuffle = True, batch_size = 64)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (224, 224, 3)))

model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Flatten())

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))



model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
model.fit_generator(trn_batches, steps_per_epoch = trn_batches.n / 128, epochs = 1, validation_data = val_batches, validation_steps = val_batches.n / 128)
model.save_weights('model_weight.h5')
df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

df.head()
!unzip -q ../input/dogs-vs-cats-redux-kernels-edition/test.zip
!mkdir test/unk/
!mv test/*.jpg test/unk/
test = datagen.flow_from_directory('test/', target_size=(224, 224), batch_size=64)
results = model.predict_generator(test)
results
filenames = [file.split('/')[1].replace('.jpg', '') for file in test.filenames]

filenames[:2]
results_df = pd.DataFrame(data={'id': filenames, 'label': list(results[:, 1])})

results_df.head()
results_df.to_csv("submissions.csv", index=False)
from IPython.display import FileLink



FileLink('submissions.csv')