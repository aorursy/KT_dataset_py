# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import random

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 150

COLS = 150

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:1000] + train_cats[:1000]

validation = train_dogs[1000:1400] + train_cats[1000:1400]

random.shuffle(train_images)

test_images =  test_images[:25]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image

    #print(data[1].shape)

    return data



train = prep_data(train_images)

test = prep_data(test_images)

validation = prep_data(validation)

labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)

print(labels)
img_width, img_height = 150, 150

nb_train_samples = 2000

nb_validation_samples = 800

epochs = 50

batch_size = 16

input_shape = (img_width, img_height, 3)







model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape,border_mode='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3),border_mode='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3),border_mode='same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))



model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# this is the augmentation configuration we will use for training

"""train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(train ,batch_size=batch_size)



validation_generator = test_datagen.flow(validation , batch_size=batch_size)"""

"""model.fit_generator(

        train_generator,

        steps_per_epoch=2000 // batch_size,

        epochs=50,

        validation_data=validation_generator,

        validation_steps=800 // batch_size)

"""

model.fit(train, labels, batch_size=batch_size, epochs=epochs)
predictions = model.predict(test, verbose=0)

print(predictions)

c=0

for i in range(0,len(predictions)):

    if predictions[i]==0:

        c=c+1

print(c)