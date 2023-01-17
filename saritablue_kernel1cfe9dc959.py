# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#packages required to build CNN

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
#initializing the CNN

classifier = Sequential()
#adding different layers

#Step 1 :Convolution

classifier.add(Convolution2D(64, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
#step 2 : pooling

#classifier.add(MaxPooling2D(pool_size = (2,2)))

#also check the following command

classifier.add(MaxPooling2D(pool_size = (2,2), strides = 2))
#adding another con layer to obtain better accuracy

#32 is number of filters (3,3): size of filter

#classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))
#step 3: Flattening

classifier.add(Flatten())
#step 4: Full Connection

#hidden layer

classifier.add(Dense(units = 256, activation = 'relu'))



#output layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))
#compile neural net

classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
#part 2: Fitting the CNN to images

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.0,

        zoom_range=0.0,

        horizontal_flip=False)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory('/kaggle/input/dataset/training_set',target_size=(256, 256),batch_size=32,class_mode='binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/dataset/test_set',target_size=(256, 256),batch_size=32,class_mode='binary')



classifier.fit_generator(training_set, steps_per_epoch=1500,epochs=20,validation_data=test_set,validation_steps=500)