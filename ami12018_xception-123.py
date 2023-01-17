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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.callbacks import CSVLogger, ModelCheckpoint

training_data_dir= '/kaggle/input/leaf-classification/dataset/train'

validation_data_dir='/kaggle/input/leaf-classification/dataset/test'

IMAGE_SIZE = 224

IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE

EPOCHS = 25

BATCH_SIZE = 64

TEST_SIZE = 30

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

import tensorflow as tf

import tensorflow.keras



from tensorflow.keras import models, layers

from tensorflow.keras.models import Model, model_from_json, Sequential



from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D



from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import SGD, RMSprop

from tensorflow.keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

training_data_generator = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True,

    rotation_range=60 ,

    brightness_range=[0.5 , 1.0]

    

    )

validation_data_generator = ImageDataGenerator(rescale=1./255,

         horizontal_flip=True,

    vertical_flip=True,

    rotation_range=60,

    brightness_range=[0.5 , 1.0])

training_generator = training_data_generator.flow_from_directory(

    training_data_dir,

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    shuffle = True,

    class_mode="categorical")



validation_generator = validation_data_generator.flow_from_directory(

    validation_data_dir,

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    batch_size=BATCH_SIZE,

    shuffle = True,

    class_mode="categorical")

def entry_flow(inputs) :



    x = Conv2D(32,(3,3), strides = 2, padding='valid')(inputs)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)



    x = Conv2D(64,(3,3),padding='same')(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x=Dropout(0.2)(x)

    previous_block_activation = x



    for size in [128, 256, 728] :



        x = Activation('relu')(x)

        x = SeparableConv2D(size, 3, padding='same')(x)

        x = BatchNormalization()(x)



        x = Activation('relu')(x)

        x = SeparableConv2D(size, 3, padding='same')(x)

        x = BatchNormalization()(x)



        x = MaxPooling2D(3, strides=2, padding='same')(x)

        x=Dropout(0.2)(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)



        x = tensorflow.keras.layers.Add()([x, residual])

        previous_block_activation = x



    return x

def middle_flow(x, num_blocks=8) :



    previous_block_activation = x



    for _ in range(num_blocks) :



        x = Activation('relu')(x)

        x = SeparableConv2D(728, (3,3), padding='same')(x)

        x = BatchNormalization()(x)



        x = Activation('relu')(x)

        x = SeparableConv2D(728,( 3,3), padding='same')(x)

        x = BatchNormalization()(x)

       

        x = Activation('relu')(x)

        x = SeparableConv2D(728, (3,3), padding='same')(x)

        x = BatchNormalization()(x)

        x=Dropout(0.2)(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation])

        previous_block_activation = x



    return x

def exit_flow(x) :



    previous_block_activation = x



    x = Activation('relu')(x)

    x = SeparableConv2D(728,( 3,3), padding='same')(x)

    x = BatchNormalization()(x)

    

    x = Activation('relu')(x)

    x = SeparableConv2D(1024, (3,3), padding='same')(x) 

    x = BatchNormalization()(x)



    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x=Dropout(0.2)(x)

    residual = Conv2D(1024, (1,1), strides=2, padding='same')(previous_block_activation)

    x = tensorflow.keras.layers.Add()([x, residual])



    x = Activation('relu')(x)

    x = SeparableConv2D(728, (3,3), padding='same')(x)

    x = BatchNormalization()(x)



    x = Activation('relu')(x)

    x = SeparableConv2D(1024,(3,3), padding='same')(x)

    x = BatchNormalization()(x)

    x=Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)

    

    x = Dense(185, activation='softmax')(x)



    return x

inputs = Input(shape=(224, 224, 3))

outputs = exit_flow(middle_flow(entry_flow(inputs)))

xception = Model(inputs, outputs)

#model=xception.summary()

from tensorflow.keras.callbacks import EarlyStopping

#es=tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1)

xception.compile(optimizer = tensorflow.keras.optimizers.Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])







history=xception.fit_generator(

    training_generator,

    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,

    epochs=15,

    validation_data=validation_generator,

    validation_steps=len(validation_generator.filenames) // BATCH_SIZE)











xception.compile(optimizer = tensorflow.keras.optimizers.Adam(lr = 0.001),

              loss='categorical_crossentropy', metrics=['accuracy',])

history=xception.fit_generator(

    training_generator,

    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,

    epochs=15,

    validation_data=validation_generator,

    validation_steps=len(validation_generator.filenames) // BATCH_SIZE

           )



from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.h5',monitor = 'val_loss', verbose=1, save_best_only=True)

callbacks = [ checkpointer]



xception.compile(optimizer = tensorflow.keras.optimizers.Adam(lr = 0.0001),

              loss='categorical_crossentropy', metrics=['accuracy',])

history=xception.fit_generator(

    training_generator,

    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,

    epochs=5,

    validation_data=validation_generator,

    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,

   callbacks = [ checkpointer]        )
