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
#using CNN to do image classification

#import required Keras packages 



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

#Building the CNN



#initializing the network

classifier = Sequential()



#step-1 : add convolution layer

classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))



#step-2 : add max pooling layer

classifier.add(MaxPooling2D(pool_size=(2,2)))



# add a second convolution layer and a max pooling layer

classifier.add(Conv2D(32,(3,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



#step-3 Flattening

classifier.add(Flatten())



#step-4 : add a Fully connected Layer

classifier.add(Dense(units=128 ,activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))



#compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Loading and Processing the images

#use ImageGenerator to augment the images



from keras.preprocessing.image import ImageDataGenerator



train_data = "/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set"

test_data = "/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set"





train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)

### flow_from_directory

## directory : string, path to the target directory. It should contain one subdirectory per class. 

### Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator

# source : keras documentation - https://keras.io/preprocessing/image/



train_generator = train_datagen.flow_from_directory(

        train_data,

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

    test_data,

    target_size = (64, 64),

    batch_size = 32,

    class_mode = 'binary')

## Fit the model to the CNN



classifier.fit_generator(

        train_generator,        

        epochs=25,

        validation_data=test_set

        )