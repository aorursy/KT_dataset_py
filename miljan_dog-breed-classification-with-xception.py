import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model



        

import keras.backend as K

K.set_image_data_format('channels_last')



from matplotlib.pyplot import imshow

from keras.preprocessing import image

from keras import applications

from keras.models import Sequential

import os,sys

import warnings

warnings.simplefilter("ignore")



import cv2

from keras.preprocessing.image import ImageDataGenerator
os.listdir('../input/cropped')
img = cv2.imread('../input/cropped/cropped/train/n02085620-Chihuahua/n02085620_7.jpg')

print(img.shape)

img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img1)
train_dir = '../input/cropped/cropped/train/'

test_dir = '../input/cropped/cropped/test/'
# Training generator

datagen = ImageDataGenerator( 

    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)

    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)

    rotation_range=40,

    zoom_range = 0.1,

    rescale=1./255,

    validation_split=0.33)

train_generator = datagen.flow_from_directory(

    directory=train_dir,

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=64,

    class_mode="categorical",

    shuffle=True,

    seed=42, 

    subset="training"

)



# Valid generator



valid_generator = datagen.flow_from_directory(

    directory=train_dir,

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=64,

    class_mode="categorical",

    shuffle=True,

    seed=42, 

    subset="validation"

)



# Test generator



test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(

    directory=test_dir,

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=32,

    class_mode="categorical",

    shuffle=False,

    seed=42

)
from keras import optimizers

from keras.callbacks import ReduceLROnPlateau



optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
from keras.applications.xception import Xception

base_model = Xception(weights='imagenet', include_top=False)
# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- 

predictions = Dense(120, activation='softmax')(x)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional Xception layers

for layer in base_model.layers:

    layer.trainable = False

    

# this is the model we will train

xception = Model(inputs=base_model.input, outputs=predictions)
xception.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
xception.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=10,

                    verbose=1,

                    callbacks=[learning_rate_reduction]

)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
loss, acc = xception.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print(loss, acc)