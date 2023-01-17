import tensorflow as tf

import matplotlib.pyplot as plt 

from matplotlib.image import imread 

import os
IMAGE_SIZE = 224

CHANNELS = 3

DATADIR = '../input/chest-xray-pneumonia/chest_xray/'

test_path = DATADIR + '/test/'

valid_path = DATADIR + '/val/'

train_path = DATADIR + '/train/'

BATCH_SIZE = 32

CATEGORIES = ["NORMAL", "PNEUMONIA"]
if CHANNELS == 1:

    color_mode = "grayscale"

elif CHANNELS == 3:

    color_mode = "rgb"
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255)

train_images = train_datagen.flow_from_directory(train_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode="binary",classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)
test_datagen = ImageDataGenerator(rescale=1/255)

test_images = test_datagen.flow_from_directory(test_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode="binary",classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)
base_model = tf.keras.applications.DenseNet121(weights=None,include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(64,activation='relu')(x)

x = tf.keras.layers.Dense(32,activation='relu')(x)

x = tf.keras.layers.Dense(1,activation='sigmoid')(x)



my_model = tf.keras.Model(inputs=base_model.input,outputs=x)

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),loss='binary_crossentropy',metrics=['accuracy'])
my_model.fit_generator(train_images,validation_data=test_images,epochs=20)