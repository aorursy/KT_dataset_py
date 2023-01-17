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
#put test images in a vector

import os

from tensorflow.keras.preprocessing import image



test_images_vector = []

label_test = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if('train/predator' in dirname):

            images = image.load_img(os.path.join(dirname, filename), target_size = (50, 50)) 

            test_images = image.img_to_array(images)

            test_images_vector.append(test_images)

            label_test.append(0)

        elif('train/alien' in dirname):

            images = image.load_img(os.path.join(dirname, filename), target_size = (50, 50)) 

            test_images = image.img_to_array(images)

            test_images_vector.append(test_images)            

            label_test.append(1)



test_images_vector = np.asarray(test_images_vector)

label_test = np.asarray(label_test)
from random import randrange



def generator(features, labels, batch_size):

     # Create empty arrays to contain batch of features and labels#

     batch_features = np.zeros((batch_size, 50, 50, 3))

     batch_labels = np.zeros((batch_size,1))

     while True:

       for i in range(batch_size):

         # choose random index in features

         index= randrange(1,len(features))

         batch_features[i] = features[index]

         batch_labels[i] = labels[index]

       yield batch_features, batch_labels
import keras

from keras.models import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

from keras.optimizers import adam

import numpy as np
# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3)))

classifier.add(Activation("relu"))

classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Conv2D(64, (3, 3), input_shape = (100, 100, 3)))

classifier.add(Activation("relu"))

classifier.add(MaxPooling2D(pool_size = (3, 3)))



classifier.add(Flatten())



classifier.add(Dense(64))

classifier.add(Activation("relu")) 

classifier.add(Dense(128))

classifier.add(Activation("relu")) 

classifier.add(Dense(activation = 'sigmoid', units=1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()
import random

classifier.fit_generator(generator(test_images_vector, label_test, 40), steps_per_epoch=50, epochs=10)