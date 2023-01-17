# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator #hehe xd

from tensorflow.keras.applications import mobilenet_v2, vgg19

import cv2
TRAIN_DIR = '/kaggle/input/shopee-product-detection-student/train/train/train'

TEST_DIR = '/kaggle/input/shopee-product-detection-student/test/test' # One directory before the individual images for ImageDataGenerator

IMG_SIZE = 224 # Smallest size for transfer learning approach

BATCH_SIZE = 128

CLASSES = 42

SPLIT=0.15

TOTAL_IMAGE = 105390
train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=SPLIT) # set validation split
train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training')
validation_generator = train_datagen.flow_from_directory(

    TRAIN_DIR, # same directory as training data

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation')
base_model = tf.keras.applications.MobileNet(input_shape=(IMG_SIZE,IMG_SIZE,3),

                                               include_top=False,

                                               pooling='avg',

                                               weights='imagenet')

# fine tune with just 1 untrained layer at a time

# trial and error approach but untraining the layer by layer

mid_start = base_model.get_layer('conv_pw_12')

for i in range(base_model.layers.index(mid_start)):

    base_model.layers[i].trainable = False

    

print("Set trainable layers")
base_model.summary()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D



model = Sequential([

    base_model,

    Dense(CLASSES, activation=tf.nn.softmax)

    

])



model.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer=tf.keras.optimizers.Adam(),

              metrics=['accuracy'])



print("model compiled")
model.summary()
from tensorflow.python.client import device_lib



sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print(sess)
TRAIN_SAMPLES = TOTAL_IMAGE * (1 - SPLIT)

VALIDATION_SAMPLES = TOTAL_IMAGE * SPLIT

EPOCHS = 6



model.fit(

    train_generator,

    steps_per_epoch = TRAIN_SAMPLES // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = VALIDATION_SAMPLES // BATCH_SIZE,

    epochs = EPOCHS)
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

        TEST_DIR,

        target_size=(IMG_SIZE, IMG_SIZE),

        color_mode="rgb",

        shuffle = False,

        class_mode=None,

        batch_size=1)



filenames = test_generator.filenames

nb_samples = len(filenames)

np_filename = np.array(filenames)
predict = model.predict(test_generator,steps = nb_samples) # model output seems poor, can attempt to improve for later versions

#predict = [[1, 3, 3,6 ,5, 6] for x in range(0, 12191)]

#predict = np.array(predict)

result = pd.DataFrame([np_filename, predict]).transpose()



result.columns = ['filename', 'category']

result['filename'] = result['filename'].apply(lambda val: val.split('/')[1])

result['category'] = result['category'].apply(lambda val: np.array(val).argmax())

result['category'] = result["category"].apply(lambda x: "{:02}".format(x))

testDF = pd.read_csv('/kaggle/input/shopee-product-detection-student/test.csv')

final_ans = testDF.drop(columns='category').merge(result, on='filename', how='left')

final_ans.to_csv('/kaggle/working/results.csv', index=False)

final_ans.head(10)