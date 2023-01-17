import os, json

from glob import glob

import numpy as np

import scipy

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4, linewidth=100)

from matplotlib import pyplot as plt



from numpy.random import random, permutation

from scipy import misc, ndimage

from scipy.ndimage.interpolation import zoom



import keras

from keras import backend as K

from keras.utils.data_utils import get_file

from keras.models import Sequential

from keras.layers import Input

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD, RMSprop

from keras.preprocessing import image

from keras import applications

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D 

from keras.layers.normalization import BatchNormalization

model = Sequential()

   

model.add(Conv2D(16, 3, input_shape = (32, 32, 3), activation = 'relu'))

       

model.add(Conv2D(16, 3,  activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))



model.add(Conv2D(32, 3,  activation = 'relu'))       

model.add(Conv2D(32, 3,  activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 







model.add(Flatten())





model.add(Dense(10, activation='softmax'))

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.1,

                                   zoom_range = 0.1,

                                               )



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('train',

                                                 target_size = (32, 32),

                                                 batch_size = 4,

                                                 class_mode = 'categorical')



test_set = test_datagen.flow_from_directory('val',

                                            target_size = (32, 32),

                                            batch_size = 4,

                                            class_mode = 'categorical')
model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), loss='categorical_crossentropy', metrics=['accuracy'])



model.fit_generator(training_set,

                         steps_per_epoch = 22400/4,

                         epochs = 2,

                         validation_data = test_set,

                         validation_steps = 4000/4)