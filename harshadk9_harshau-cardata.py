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
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
classifier=Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=190,activation='relu'))

classifier.add(Dense(output_dim=189,activation='softmax'))

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_set = train_datagen.flow_from_directory(

        '../input/vehicledetected-stanford-cars-data-classes-folder/stanford-car-dataset-by-classes-folder/stanford-car-dataset-by-classes-folder/car_data/car_data/train',

        target_size=(64,64),

        batch_size=32,

        class_mode='categorical')



test_set = test_datagen.flow_from_directory(

        '../input/vehicledetected-stanford-cars-data-classes-folder/stanford-car-dataset-by-classes-folder/stanford-car-dataset-by-classes-folder/car_data/car_data/test',

        target_size=(64,64),

        batch_size=32,

        class_mode='categorical')

classifier.fit_generator(

    	train_set,

    	steps_per_epoch=8144,

    	epochs=5,

    	validation_data=test_set,

    	validation_steps=8041)