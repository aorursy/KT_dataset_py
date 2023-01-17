import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data
import pandas as pd

import numpy as np
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

import keras

K.set_image_dim_ordering('th')
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))



lst_prod = []

for c, d in enumerate(data):

    lst_prod.append(d['category_id'])
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



y = lst_prod



encoder = LabelEncoder()

encoder.fit(y)

encoded_y = encoder.transform(y)

dummy_y = np_utils.to_categorical(encoded_y)

dummy_y.shape
num_classes=len(dummy_y[81])

epochs = 5
model = Sequential()

# Convolutional Layer

model = Sequential()

# Convolutional Layer

model.add(Conv2D(180, (3,3), input_shape = (180,180,3), activation='relu'))



# Pooling Layer

model.add(MaxPooling2D(pool_size=(1, 1)))



# Fully conected Layer

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))





lrate = 0.01

decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)



model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))



prod_to_category = dict()

i=0

for c, d in enumerate(data):

    

    lst_pic = []

    for e, pic in enumerate(d['imgs']):

        picture = imread(io.BytesIO(pic['picture']))

        

        picture = picture.reshape(1,180,180,3)

        # do something with the picture, etc

#         print(picture.shape)

        lst_pic.append(picture)



    # train on single row

    for j in lst_pic:

        X_batch = j

        Y_batch = dummy_y[i]

        Y_batch = Y_batch.reshape(1,num_classes)

        model.fit(X_batch, Y_batch, batch_size=32, epochs=epochs)

    i = i+1
# picture.reshape(1,180,180,3)