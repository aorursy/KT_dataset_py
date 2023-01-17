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
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/urban-and-rural-photos/rural_and_urban_photos/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/urban-and-rural-photos/rural_and_urban_photos/val',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        validation_data=validation_generator,

        validation_steps=1)
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Indicate whether the first layer should be trained/changed or not.

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocess_input)



train_generator = data_generator.flow_from_directory(

                                        directory= '../input/dogs-gone-sideways/images/train',

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

                                        directory='../input/dogs-gone-sideways/images/val',

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')



# fit_stats below saves some statistics describing how model fitting went

# the key role of the following line is how it changes my_new_model by fitting to data

fit_stats = my_new_model.fit_generator(train_generator,

                                       steps_per_epoch=22,

                                       validation_data=validation_generator,

                                       validation_steps=1)
