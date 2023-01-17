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
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
classifier = Sequential()
classifier.add(Convolution2D(filters=32, kernel_size = (3,3) , input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128 , activation = 'relu'))
classifier.add(Dense(units = 1 , activation = 'sigmoid'))
classifier.compile(optimizer= 'adam' , loss = 'binary_crossentropy' , metrics= ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('../input/cat-and-dog/training_set/training_set',

                                                    target_size=(64, 64),

                                                    batch_size=32,

                                                    class_mode='binary')
test_set = test_datagen.flow_from_directory('../input/cat-and-dog/test_set/test_set',

                                            target_size=(64, 64),

                                            batch_size=32,

                                            class_mode='binary')