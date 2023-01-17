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
weights_path = '../input/VGG-16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D

from keras.optimizers import Adam

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16





img_width, img_height = 224, 224



train_data_dir = '../input/flowers-recognition/flowers/flowers/'

nb_train_samples = 3462

nb_validation_samples = 861

epochs = 10

batch_size = 32



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



input_shape
# create the base pre-trained model

base_model = VGG16(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(5, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False



model.compile(loss ='categorical_crossentropy',optimizer ='adam', metrics =['accuracy'])



data_gen = ImageDataGenerator(rescale = 1. / 255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,validation_split=0.2)



train_generator = data_gen.flow_from_directory(train_data_dir,target_size =(img_width, img_height),seed=13,

                                                    batch_size = batch_size, class_mode ='categorical',subset="training")



validation_generator = data_gen.flow_from_directory(train_data_dir,target_size =(img_width, img_height),seed=13,

                                                        batch_size = batch_size, class_mode ='categorical',subset="validation")



model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size,

                    epochs = epochs, validation_data = validation_generator,

                    validation_steps = nb_validation_samples // batch_size)
