# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) no need
import tensorflow as tf
print(tf.__version__)
import tensorflow_hub as tfhub

import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAIN_DIR = "/kaggle/input/fruits/fruits-360/Training/"
VAL_DIR = "/kaggle/input/fruits/fruits-360/Test/"

print(len(os.listdir(TRAIN_DIR)))
print(len(os.listdir(VAL_DIR)))
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                  vertical_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (227, 227, 3))     

# Flow validation images in batches of 32 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(VAL_DIR,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (227, 227, 3))
feature_map  =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11,11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
    tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),
    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),
    tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(4096, activation='relu'),
#     tf.keras.layers.Dense(4096, activation='relu'),
#     tf.keras.layers.Dense(1000, activation='softmax')
])

NUM_ANCHORS = 9
rpn = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(feature_map.output)
X_class = tf.keras.layers.Conv2D(2*NUM_ANCHORS, (1, 1), activation='softmax')(rpn)
X_regr = tf.keras.layers.Conv2D(4*NUM_ANCHORS, (1, 1))(rpn)

#Add ROI Pooling

fc = tf.keras.layers.Flatten()(roi_pool)
fc1 = tf.keras.layers.Dense(4096, activation='relu')(fc)
fc2 = tf.keras.layers.Dense(4096, activation='relu')(fc1)

cls_score = tf.keras.layers.Dense(NUM_CLASSES, activation='relu')

bbox_pred = tf.keras.layers.Dense(4*NUM_CLASSES)

model = tf.keras.Model(feature_map.input, output)
model.summary()
model.output
