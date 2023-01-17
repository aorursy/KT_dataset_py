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

!nvidia-smi

!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import models,layers
print(tf.__version__)
!pip install tensorflow_datasets

import tensorflow_datasets as tfds
(train,val,test),metadata = tfds.load("cats_vs_dogs",split=list(tfds.Split.TRAIN.subsplit(weighted=(8,1,1))),as_supervised=True,with_info=True)
image_size = 160

def preprocess_image(image,label):

    im = tf.cast(image,tf.float32)

    im = im/127.5-1

    im = tf.image.resize(im,(image_size,image_size))

    return im,label
train = train.map(preprocess_image)

val = val.map(preprocess_image)

test = test.map(preprocess_image)
train = train.shuffle(1000).batch(32)

val = val.batch(32)

test = test.batch(32)
for image_batch, label_batch in train.take(1):

    print(image_batch.shape)

    print(label_batch.shape)
base_model = tf.keras.applications.MobileNetV2(input_shape=(image_size,image_size,3),include_top=False,weights="imagenet")
feature = base_model(image_batch)

print(feature.shape)
average_pooling = layers.GlobalAveragePooling2D()

feature_average = average_pooling(feature)

print(feature_average.shape)
prediction_layer = layers.Dense(1)

prediction = prediction_layer(feature_average)

print(prediction.shape)
base_model.trainable=False
model = models.Sequential([base_model,average_pooling,prediction_layer])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

initial_epochs = 5

len(model.trainable_variables)
num_train, num_val, num_test = (metadata.splits['train'].num_examples * weights/10 for weights in (8,1,1))

print(num_train,num_val, num_test)
steps_per_epoch = int(num_train/32)

validation_steps = int(num_val/32)
history = model.fit(train.repeat(),epochs=initial_epochs,steps_per_epoch=steps_per_epoch,

                    validation_data=val.repeat(),validation_steps=validation_steps)
fine_tune = 100

base_model.trainable = True

for layer in base_model.layers[:fine_tune]:

    layer.trainable = False
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00001),loss='binary_crossentropy',

             metrics=['accuracy'])

fine_tune_epochs = 5

total_epochs = initial_epochs + fine_tune_epochs
history_tune = model.fit(train.repeat(),epochs=total_epochs,steps_per_epoch=steps_per_epoch,

                         initial_epoch=initial_epochs,validation_data=val.repeat(),

                        validation_steps=validation_steps)