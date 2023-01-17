import itertools

import os

import pandas as pd



import matplotlib.pylab as plt

import numpy as np



import tensorflow as tf

import tensorflow_hub as hub

# CONSTANT



train_img_path = '../input/shopee-round-2-product-detection-challenge/train/train'

test_img_path = '../input/shopee-round-2-product-detection-challenge/test'

img_width = 224

img_height = 224



batch_size = 16
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.05,shear_range=0.5,

                                                            zoom_range=0.5,rotation_range=90,horizontal_flip=True,vertical_flip=True)

train_data_gen = train_gen.flow_from_directory(batch_size=batch_size,

                                                           directory=train_img_path,

                                                           shuffle=True,

                                                           target_size=(img_height, img_width),

                                                           subset="training")

val_data_gen = train_gen.flow_from_directory(batch_size=batch_size,

                                                           directory=train_img_path,

                                                           shuffle=True,

                                                           target_size=(img_height, img_width),

                                                           subset="validation")
# feature_extractor_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4" 

# feature_extractor_layer = hub.KerasLayer(feature_extractor_url,

#                                          input_shape=(img_width,img_height,3))

# feature_extractor_layer.trainable = False
# model = tf.keras.Sequential([

#   feature_extractor_layer,

#   tf.keras.layers.Dense(42,activation="softmax")

# ])



# model.summary()



model = tf.keras.models.load_model('../input/shopee/model.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

model.summary()
model.compile(

  optimizer=tf.keras.optimizers.Adam(),

  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

  metrics=['acc'])
model.fit(x=train_data_gen,

            epochs=3,

            validation_data=val_data_gen)



# model.fit(x=train_data_gen,

#             epochs=10,

#             validation_data=val_data_gen)
model.save('/kaggle/working/model.h5')
f = open("/kaggle/working/v.txt", "a")

f.write("20")

f.close()