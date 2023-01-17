import pathlib

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.models as km

import tensorflow.keras.layers as kl

import matplotlib.pyplot as plt

%matplotlib inline

import IPython.display as display

tf.enable_eager_execution()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/sample_submission.csv')

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_names = train['id']

ytrain = train['has_cactus']
train_image_names.values
train_image_paths = '../input/train/train/'+ train_image_names.values
train_image_paths
image = train_image_paths[0]
ytrain.values
def preprocess_image(image):

    image = tf.image.decode_jpeg(image,channels=3)

    image = tf.image.resize_images(image,[32,32])

    image /= 255.0

    return image
def load_and_preprocess_image(path):

    image = tf.read_file(path)

    return preprocess_image(image)
path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
labels_ds = tf.data.Dataset.from_tensor_slices(ytrain)
ds_label_ds = tf.data.Dataset.zip((ds,labels_ds))
ds_label_ds = ds_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(train)))
ds_label_ds = ds_label_ds.batch(30)

ds_label_ds = ds_label_ds.prefetch(buffer_size=AUTOTUNE)
ds_label_ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = km.Sequential([

    kl.Conv2D(filters=32, kernel_size=3,padding='same',input_shape=(32,32,3),activation=tf.nn.relu),    

])
model.add(kl.Conv2D(32, (3, 3)))

model.add(kl.Activation('relu'))

model.add(kl.MaxPooling2D(pool_size=(2, 2)))

model.add(kl.Dropout(0.25))



model.add(kl.Conv2D(64, (3, 3), padding='same'))

model.add(kl.Activation('relu'))

model.add(kl.Conv2D(64, (3, 3)))

model.add(kl.Activation('relu'))

model.add(kl.MaxPooling2D(pool_size=(2, 2)))

model.add(kl.Dropout(0.25))



model.add(kl.Conv2D(64, (3, 3), padding='same'))

model.add(kl.Activation('relu'))

model.add(kl.Conv2D(64, (3, 3)))

model.add(kl.Activation('relu'))

model.add(kl.MaxPooling2D(pool_size=(2, 2)))

model.add(kl.Dropout(0.25))



model.add(kl.Flatten())

model.add(kl.Dense(512))

model.add(kl.Activation('relu'))

model.add(kl.Dropout(0.5))

model.add(kl.Dense(2))

model.add(kl.Activation('softmax'))
model.compile(optimizer='adam',loss=tf.keras.losses.sparse_categorical_crossentropy,

             metrics=['accuracy'])
model.fit(ds_label_ds,epochs=5,steps_per_epoch=len(train)//5)
test.shape
test_image_names = test['id']
test_image_paths = '../input/test/test/'+test_image_names
test_image_paths.values
Xtest = []
import cv2
for path in test_image_paths:

    image = cv2.imread(path)

    Xtest.append(image)
Xtest = np.reshape(Xtest,newshape=(-1,32,32,3))

Xtest = Xtest / 255.0
test_ds = tf.data.Dataset.from_tensor_slices(Xtest)
test_ds = test_ds.batch(30)
pre = model.predict(test_ds,steps=len(test))
pre.shape
pre_ = np.argmax(pre,axis=1)
pre_
test.has_cactus = pre_
test.to_csv('submission_7.csv',index=False)