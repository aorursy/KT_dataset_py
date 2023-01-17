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
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.layers.normalization import BatchNormalization

from keras.utils import multi_gpu_model

from keras.callbacks.callbacks import EarlyStopping,ModelCheckpoint

from keras.preprocessing import image

from keras import applications

import numpy as np
# train_datagen = ImageDataGenerator(

#         rescale=1./255,

#         rotation_range=20,

#         width_shift_range=0.2,

#         height_shift_range=0.2,

#         shear_range=0.2,

#         zoom_range=0.2,

#         horizontal_flip=True)



image_size = 256

batch_size = 32



train_datagen = ImageDataGenerator(

    rescale=1./255,

    featurewise_center=True,

    samplewise_center=True,

        width_shift_range=0.2,

        height_shift_range=0.2,

    validation_split=0.2,

horizontal_flip=True) # set validation split



train_generator = train_datagen.flow_from_directory(

    '/kaggle/input/gchannel/gchannel/',

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='binary',

    subset='training') # set as training data



validation_generator = train_datagen.flow_from_directory(

    '/kaggle/input/gchannel/gchannel/' ,# same directory as training data

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='binary',

    subset='validation')
img_height = img_width = 256

num_classes = 1
base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (256,256,3))

from keras import models, layers

from keras.layers import Input, Dense, MaxPooling2D

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda

from keras.models import Model

from keras.layers import BatchNormalization


model = models.Sequential()

model.add(base_model)

# model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Dropout(0.25))

# model.add(Dropout(0.5))

model.add(Conv2D(256, (3,3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(layers.Flatten())

# model.add(layers.Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()



from keras.optimizers import SGD, Adam

sgd = SGD(lr=0.0001, momentum=0.9, decay=0.01, nesterov=False)

adam = Adam(lr=0.0001)

# model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
es =  EarlyStopping(monitor='val_loss',patience=50, mode='min', verbose=1)

# es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)

# cs =  ModelCheckpoint('g_channel_loss.h5',monitor='val_loss', save_best_only=True,save_weights_only=False)

cs =  ModelCheckpoint('g_channel_accuracy.h5',monitor='val_accuracy', save_best_only=True,save_weights_only=False)

model.fit_generator(train_generator,nb_epoch=100,steps_per_epoch = 220, validation_data=validation_generator, validation_steps = 55, callbacks=[es,cs] ) # nb_val_samples=32, 