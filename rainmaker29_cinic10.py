import tensorflow as tf

import tensorflow_hub as hub



import tensorflow_datasets as tfds



import time



from PIL import Image

import requests

from io import BytesIO



import matplotlib.pyplot as plt

import numpy as np



import os

import pathlib





from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import Callback

from IPython.display import SVG,Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAIN_DIR = '../input/cinic-private/cinic-10_image_classification_challenge-dataset/cinic-10_image_classification_challenge-dataset/train'

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255,rotation_range=30,

                             zoom_range=0.15,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             shear_range=0.15,

                             horizontal_flip=True,

                             fill_mode="nearest")



train_generator = datagen.flow_from_directory(

    TRAIN_DIR, 

    subset='training',

    batch_size=128

)



val_generator = datagen.flow_from_directory(

    TRAIN_DIR,

    subset='validation',

    batch_size=64

)
# Load model fine-tuned on ImageNet

model_url = "https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1"

imagenet_module = hub.KerasLayer(model_url)
imagenet_module.trainable = True
model = tf.keras.models.Sequential([

                                    tf.keras.layers.InputLayer((32,32,3)),

                                    imagenet_module,

                                    tf.keras.layers.Dense(1024),

                                    tf.keras.layers.Dense(512),

                                    tf.keras.layers.Dense(256),

                                    tf.keras.layers.Dense(10,activation='softmax')

])

model.load_weights('../input/cinic-private/model_weights(3).h5')
import tensorflow.keras.backend as K



def f1(y_true, y_pred):

    y_true = K.flatten(y_true)

    y_pred = K.flatten(y_pred)

    return 2 * (K.sum(y_true * y_pred)+ K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


BATCH_SIZE = 128

lr = 0.003 * BATCH_SIZE / 512 

# lr = 0.003 



# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200, 300, 400], 

                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)



model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy',f1])


from keras import callbacks 

# earlystopping = callbacks.EarlyStopping(monitor ="val_loss",  

#                                         mode ="min", patience = 5,  

#                                         restore_best_weights = True) 





# reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.1,patience=5,min_lr=0.00001,mode="auto")

# 

#Saving the model 

checkpoint = ModelCheckpoint("model_weights.h5",monitor='accuracy',

                             save_weights_only=True,

                             mode="max",

                             verbose=1

                             )



callbacks = [checkpoint]

epochs = 15

steps_per_epoch = train_generator.n//train_generator.batch_size

validation_steps = val_generator.n//val_generator.batch_size

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=steps_per_epoch,

                    epochs=epochs,

                    validation_data=val_generator,

                    validation_steps=validation_steps,

                    callbacks=callbacks)
model.save('cinic_biggt.h5')
# from IPython.display import FileLink

# FileLink(r'./model_weights.h5')