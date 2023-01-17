import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



classifier=Sequential()

classifier.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3)))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation="relu", kernel_initializer="uniform"))

classifier.add(Dense(output_dim=1,activation='sigmoid',kernel_initializer="uniform"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('../input/dataset/dataset/training_set',

                                             target_size=(64,64),

                                             batch_size=32,

                                            class_mode='binary')

test_set=train_datagen.flow_from_directory('../input/dataset/dataset/test_set',

                                             target_size=(64,64),

                                              batch_size=32,

                                              class_mode='binary')



classifier.fit_generator(training_set,

                         samples_per_epoch=8000,

                         nb_epoch=25,

                         validation_data=test_set,

                         nb_val_samples=2000)

import os

print(os.listdir("../input/dataset/dataset/test_set"))