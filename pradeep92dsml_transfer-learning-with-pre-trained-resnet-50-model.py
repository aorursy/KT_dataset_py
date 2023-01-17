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
from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
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