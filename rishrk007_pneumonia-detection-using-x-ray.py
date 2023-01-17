# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import cv2

from PIL import Image

from random import shuffle

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Sequential,Model

from keras.utils.data_utils import Sequence

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

%matplotlib inline
m_dir= os.listdir('../input/chest_xray/chest_xray')

print(m_dir)
train_folder= '../input/chest_xray/chest_xray/train/'

val_folder = '../input/chest_xray/chest_xray/val/'

test_folder = '../input/chest_xray/chest_xray/test/'
print(os.listdir("../input/chest_xray/chest_xray"))

print(os.listdir("../input/chest_xray/chest_xray/train"))

print(os.listdir("../input/chest_xray/chest_xray/test/"))

img_name = 'NORMAL2-IM-0588-0001.jpeg'

img_normal = load_img('../input/chest_xray/chest_xray/train/NORMAL/' + img_name)



print('NORMAL')

plt.imshow(img_normal)

plt.show()
img_name = 'person63_bacteria_306.jpeg'

img_pneumonia = load_img('../input/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)



print('PNEUMONIA')

plt.imshow(img_pneumonia)

plt.show()
# dimensions of our images.

img_width, img_height = 150, 150
train_data_dir = '../input/chest_xray/chest_xray/train'

validation_data_dir = '../input/chest_xray/chest_xray/val'

test_data_dir = '../input/chest_xray/chest_xray/test'



nb_train_samples = 5217

nb_validation_samples = 17

epochs = 20

batch_size = 16
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()
model.input
model.output
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
test_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
history=model.fit_generator(train_generator,

                            epochs=epochs,

                            validation_data=validation_generator,

                            steps_per_epoch=nb_train_samples//batch_size,

                            validation_steps=nb_validation_samples // batch_size)
nb_test_samples=624
# evaluate the model

scores = model.evaluate_generator(test_generator, steps=np.ceil(nb_test_samples/batch_size))

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
train_data_dir1 = '../input/chest_xray/chest_xray/train'

validation_data_dir1 = '../input/chest_xray/chest_xray/val'

test_data_dir1 = '../input/chest_xray/chest_xray/test'



nb_train_samples = 5217

nb_validation_samples = 17

nb_test_samples=624

epochs = 20

batch_size = 16
from keras.applications.vgg16 import VGG16
base_model = VGG16()
base_model.summary()
add_model = Sequential()

add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(1024, activation='relu'))

add_model.add(Dense(1024, activation='relu'))

add_model.add(Dense(512, activation='relu'))

add_model.add(Dense(512, activation='relu'))

add_model.add(Dense(1, activation='sigmoid'))
add_model.summary()
model1 = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model1.summary()
img_width1=224

img_height1=224
# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(

    train_data_dir1,

    target_size=(img_width1, img_height1),

    batch_size=batch_size,

    class_mode='binary')
test_generator = test_datagen.flow_from_directory(

    test_data_dir1,

    target_size=(img_width1, img_height1),

    batch_size=batch_size,

    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(

    validation_data_dir1,

    target_size=(img_width1, img_height1),

    batch_size=batch_size,

    class_mode='binary')
model1.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
history1=model1.fit_generator(train_generator,

                            epochs=epochs,

                            validation_data=validation_generator,

                            steps_per_epoch=nb_train_samples//batch_size,

                            validation_steps=nb_validation_samples // batch_size)
# evaluate the model

scores1 = model1.evaluate_generator(test_generator, steps=np.ceil(nb_test_samples/batch_size))

print("\n%s: %.2f%%" % (model1.metrics_names[1], scores1[1]*100))