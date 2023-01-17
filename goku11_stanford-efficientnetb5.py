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
!pip install tensorflow==2.3.0 -q
import tensorflow as tf
print(tf.__version__)
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
train_data_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'
validation_data_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #shear_range=0.2,
    zoom_range=0.2,
    #fill_mode = 'constant',
    #cval = 1,
    rotation_range = 5,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_data = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
efficientb5 = tf.keras.applications.efficientnet.EfficientNetB4(include_top=None, weights="imagenet", classes=196, input_shape=(224,224,3))

x= efficientb5.output
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(196, activation="softmax")(x) #The nodes equal to number of classes(breeds)

model_EfficientNetB5 = Model(inputs = efficientb5.input, outputs = predictions)

model_EfficientNetB5.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

historyB5=model_EfficientNetB5.fit(training_data, epochs=100, validation_data= test_data)
