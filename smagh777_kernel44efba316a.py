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
# importing OpenCV(cv2) module 

import cv2 
training_data=cv2.imread('/kaggle/input/dogs-cats-images/dataset/training_set')
test_set=cv2.imread('kaggle/input/dogs-cats-images/dataset/test_set')
import tensorflow as tf
# Importing the Keras libraries and packages

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#intilaizing the CNN

classifier = Sequential()
# First Convolution layer

classifier.add(Conv2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu'))
#Pooling

classifier.add(AveragePooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(AveragePooling2D(pool_size = (2, 2)))

#third convulutional 

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(rate=0.8, noise_shape=None, seed=None))
#globalpooling

classifier.add(GlobalAveragePooling2D())

#hidden layer(128 hidden layer)

#Full connection(sigmoid because of binary outcome)

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.summary()
#adam for stochastic gradient algorithm

# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#to apply real-time data augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)
#target size according to input layer

training_set = train_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/training_set',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/test_set',

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')

classification=classifier.fit_generator(training_set,

                         epochs = 25,

                         validation_data = test_set,

                         )
score=classifier.evaluate(test_set,verbose=0)

accuracy=100*score[1]



 #print test accuracy after training

print('Test accuracy:',accuracy)
prediction=classifier.predict(test_set)