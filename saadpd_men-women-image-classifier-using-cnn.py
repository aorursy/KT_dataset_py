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

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.


import keras

from keras.models import Sequential # This package is used for implenting NN as layers

from keras.layers import Convolution2D  #  Required for adding the convolution layer 

from keras.layers import MaxPooling2D  #  Required for adding the pooling layer

from keras.layers import Flatten   # Required for converting the pooled feature maps into large featured vector

from keras.layers import Dense, Activation  # Required for implementing the fully connected neural network
# Initialing the CNN

classifier = Sequential() 

#  Implementing the convolution layer

classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation='relu'))  

# no of filters =32, kernel size (3,3), the shape of input images (64,64,3) 

# Implementing the Max pooling layer

# Adding the non-linearity to the convolution layer

classifier.add(MaxPooling2D(pool_size=(2,2)))  # implement the pooling layer and reduced the size of the feature map

#  Implenting the Flatten Layer

#  Before feeding into the neural network , we flatten the featured map into hughe single dimensional vector

classifier.add(Flatten())

#  Implementing the Neural Network 

# Implementing th hidden layer

classifier.add(Dense(128))

classifier.add(Activation('relu'))

# implementing the Output layer

classifier.add(Dense(1))

classifier.add(Activation('sigmoid'))

# Compiling the CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])





from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '/kaggle/input/menwomen-classification/traindata/traindata/',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

         '/kaggle/input/menwomen-classification/testdata/testdata/',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')

history=classifier.fit_generator(

        training_set,

        steps_per_epoch=2891,

        epochs=5,

        validation_data=test_set,

        validation_steps=1330)
classifier.save_weights('classifier.h5')
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()