# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_count = 0

for cls in os.listdir('/kaggle/input/intel-image-classification/seg_test/seg_test'): 

    z = len(os.listdir('/kaggle/input/intel-image-classification/seg_test/seg_test/'+cls))

    test_count = test_count + z

test_count
train_count = 0

for cls in os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train'): 

    z = len(os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/'+cls))

    train_count = train_count + z

train_count
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  

from keras.models import Sequential  

from keras.layers import Dropout, Flatten, Dense  

from keras import applications  

from keras.utils.np_utils import to_categorical  

import matplotlib.pyplot as plt  

import math  

import cv2  

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from keras.applications.vgg16 import VGG16

vgg_16_model=VGG16()

  

TRAINING_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                        batch_size=32,

                                        class_mode='categorical',

                                        target_size=(224, 224))



VALIDATION_DIR = "/kaggle/input/intel-image-classification/seg_test/seg_test"



validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')



validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                          batch_size=16,

                                          class_mode='categorical',

                                          target_size=(224, 224))
train_generator.class_indices
model=Sequential()

for layer in vgg_16_model.layers[:-1]:

    model.add(layer)

model.summary()
for layer in model.layers[:-18]:

    layer.trainable=False
model.add(Dense(6, activation='softmax'))
from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)



checkpointer = ModelCheckpoint(filepath='weights.{epoch: 02d}-{val_loss: .2f}.hdf5', monitor="val_loss", verbose=1, 

                               save_best_only=True, mode='auto', save_weights_only=True)

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
history =   model.fit_generator(

            train_generator,

            validation_data = validation_generator,

            steps_per_epoch = 439,

            epochs = 50,

            validation_steps = 188,

            verbose = 1,

            callbacks=[checkpointer, reduce_lr])

import matplotlib.pyplot as plt

# plt.plot(history['val_loss'])

bgr = cv2.imread('/kaggle/input/intel-image-classification/seg_pred/seg_pred/10047.jpg')

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

img = cv2.resize(rgb, (224,224), interpolation = cv2.INTER_AREA)

plt.imshow(img)

pred = model.predict(np.expand_dims(img,axis=0))

output = max(pred)

pos = np.argmax(pred)

for x in class_dict:

    if class_dict[x]==pos:

        print('ITSSS A {} BOYYYYY !!!!!'.format(x.upper()))

        plt.title(x,size=30,color='red')

plt.show()

output
class_dict = train_generator.class_indices

class_dict

# !rm /kaggle/working/weights. 6- 0.26.hdf5