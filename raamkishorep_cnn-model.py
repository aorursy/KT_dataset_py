# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

                                                    '/kaggle/input/leaf-disease/Datasets',

                                                    target_size=(64, 64),

                                                    batch_size=32,

                                                    class_mode='binary')



test_set = test_datagen.flow_from_directory(

                                                    '/kaggle/input/leafdiseasetestt/Dataset',

                                                    target_size=(64, 64),

                                                    batch_size=32,

                                                    class_mode='binary')



classifier.fit_generator(

        training_set,

        steps_per_epoch=239,

        epochs=25,

        validation_data=test_set,

        validation_steps=50)
classifier.save('cnn_model.h5')


pic = cv2.imread('/kaggle/input/test-image-of-dog/dog.jpeg')

pic = cv2.resize(pic,(64,64))

pic = np.reshape(pic,[1,64,64,3])

classifier.predict_classes(pic)