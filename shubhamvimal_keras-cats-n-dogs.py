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
# Classifier

classifier = Sequential()



# Step 1 - Convolutional

classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation='relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Adding a second convolutional layer

classifier.add(Convolution2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

 

# Adding a third convolutional layer

classifier.add(Convolution2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(output_dim = 128 , activation='relu'))

classifier.add(Dense(output_dim = 1, activation='sigmoid'))



# Compiling CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting CNN to the images

from keras.preprocessing.image import ImageDataGenerator

import numpy as np



batch_size = 64



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)

 

training_set = train_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set',

                                                 target_size = (64, 64),

                                                 batch_size = batch_size,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set',

                                            target_size = (64, 64),

                                            batch_size = batch_size,

                                            class_mode = 'binary')
classifier.fit_generator(training_set,

                         steps_per_epoch=np.ceil(training_set.samples / batch_size),

                         epochs=20,

                         validation_steps=np.ceil(test_set.samples / batch_size),

                         validation_data=test_set)