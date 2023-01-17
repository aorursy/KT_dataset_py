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
import zipfile

path = "/kaggle/input/dogs-vs-cats/train.zip"

zip_ref = zipfile.ZipFile(path, 'r')

zip_ref.extractall("/kaggle/working/")

path = "/kaggle/input/dogs-vs-cats/test1.zip"

zip_ref = zipfile.ZipFile(path, 'r')

zip_ref.extractall("/kaggle/working/")

zip_ref.close()

import os

dirname = '/kaggle/working/test1/dog'

os.mkdir(dirname)

dirname = '/kaggle/working/test1/cat'

os.mkdir(dirname)

dirname = '/kaggle/working/train/dog'

os.mkdir(dirname)

dirname = '/kaggle/working/train/cat'

os.mkdir(dirname)
dirname = '/kaggle/working/validation'

os.mkdir(dirname)
dirname = '/kaggle/working/validation/cat'

os.mkdir(dirname)

dirname = '/kaggle/working/validation/dog'

os.mkdir(dirname)
import shutil



for dirname, _, filenames in os.walk('/kaggle/working/train'):

    for filename in filenames:

        if(filename[:3] == 'dog'):

            dog_path = os.path.join(dirname, filename)

            shutil.move(dog_path, "/kaggle/working/train/dog/" + filename)

        if(filename[:3] == 'cat'):

            cat_path = os.path.join(dirname, filename)

            shutil.move(cat_path, "/kaggle/working/train/cat/" + filename)





import shutil

for dirname, _, filenames in os.walk('/kaggle/working/train/cat/'):

    i = 0

    for filename in filenames:

        if(i < 1000):

            dog_path = os.path.join(dirname, filename)

            shutil.move(dog_path, "/kaggle/working/validation/cat/" + filename)

            i = i + 1

            

for dirname, _, filenames in os.walk('/kaggle/working/train/dog/'):

    i = 0

    for filename in filenames:

        if(i < 1000):

            dog_path = os.path.join(dirname, filename)

            shutil.move(dog_path, "/kaggle/working/validation/dog/" + filename)

            i = i + 1
import tensorflow as tf

import os

import zipfile

from os import path, getcwd, chdir
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Activation



from keras.optimizers import SGD



input_shape = (200, 200, 3)

def train_model():

    IMAGE_SIZE = 200

    model = Sequential()



    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))

    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])        

    from tensorflow.keras.preprocessing.image import ImageDataGenerator



    train_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = ImageDataGenerator(rescale=1/255)



    train_generator = train_datagen.flow_from_directory(

        "/kaggle/working/train/",

        target_size=(IMAGE_SIZE, IMAGE_SIZE),

        batch_size=250,

        class_mode='binary'

    )

    

    validation_generator = train_datagen.flow_from_directory(

        "/kaggle/working/validation/",

        target_size=(IMAGE_SIZE, IMAGE_SIZE),

        batch_size=50,

        class_mode='binary'

    )



    history = model.fit_generator(

         train_generator,

         epochs=100,

         #steps_per_epoch=10,

         validation_data=validation_generator,

         verbose=1

    )

    

    model.save("model_loc.h5")

    return history.history['accuracy'][-1]
train_model()