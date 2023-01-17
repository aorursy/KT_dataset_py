# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/vehicledetected-stanford-cars-data-classes-folder/stanford-car-dataset-by-classes-folder/stanford-car-dataset-by-classes-folder/car_data/car_data"))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.optimizers import RMSprop
# Initialising the CNN

classifier = Sequential()



#1st Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



#Flattening

classifier.add(Flatten())



#Step 4 - Full connection

classifier.add(Dense(activation="relu", units=2048))

classifier.add(Dense(activation = 'softmax', units=189))
# Compiling the CNN

classifier.compile(optimizer = RMSprop(lr=0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '../input/vehicledetected-stanford-cars-data-classes-folder/stanford-car-dataset-by-classes-folder/stanford-car-dataset-by-classes-folder/car_data/car_data',

        target_size=(64, 64),

        batch_size=32,

        class_mode='sparse')



test_set = test_datagen.flow_from_directory(

        '../input/vehicledetected-stanford-cars-data-classes-folder/stanford-car-dataset-by-classes-folder/stanford-car-dataset-by-classes-folder/car_data/car_data',

        target_size=(64, 64),

        batch_size=32,

        class_mode='sparse')
classifier.fit_generator(

                             training_set,

                             samples_per_epoch = 8144,

                             nb_epoch = 15,

                             validation_data = test_set,

                             nb_val_samples = 8041)