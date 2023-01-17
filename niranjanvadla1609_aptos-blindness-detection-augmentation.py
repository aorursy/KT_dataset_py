import os

import zipfile

import numpy as np

import sys

import pandas as pd

import tensorflow as tf

from keras_preprocessing.image import load_img

from keras_preprocessing.image import img_to_array

from kaggle_datasets import KaggleDatasets

from numpy import save

from numpy import asarray

from os import listdir

import matplotlib as mpl

from numpy import load

from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt

import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator







train_df=pd.read_csv('/kaggle/input/aptos-prepare-train-and-validation-set/train.csv')

validation_df=pd.read_csv('/kaggle/input/aptos-prepare-train-and-validation-set/validation.csv')
train_df.head()
fmap1=train_df['diagnosis'].value_counts()

print(fmap1)
validation_df.head()
fmap2=validation_df['diagnosis'].value_counts()

print(fmap2)
path_for_gcs=KaggleDatasets().get_gcs_path('aptos2019-blindness-detection')

print(path_for_gcs)
train_files_path=[path_for_gcs+'/train_images/'+ fname for fname in train_df['id_code']]

validation_files_path=[path_for_gcs+'/train_images/'+ fname for fname in validation_df['id_code']]

train_labels=list(train_df['diagnosis'])

validation_labels=list(validation_df['diagnosis'])




print(train_files_path[0])

print(validation_files_path[0])



#TPU setup

try:

    tpu=tf.distribute.cluster_resolver.TPUClusterResolver()

    print("Running on TPU")

except ValueError:

    tpu=None

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    tpu_strategy=tf.distribute.experimental.TPUStrategy(tpu)

else:

    tpu_strategy=tf.distribute.get_strategy()

print("REPLICAS ",tpu_strategy.num_replicas_in_sync)




#predefined variable

IMG_WIDTH=512

IMG_HEIGHT=512

BATCH_SIZE=16*tpu_strategy.num_replicas_in_sync#take thumb rule

AUTOTUNE = tf.data.experimental.AUTOTUNE 

EPOCHS = 10

STEPS_PER_EPOCH=train_df.shape[0]//BATCH_SIZE

print(STEPS_PER_EPOCH)







def parse_function_for_train(filename,label):

    image_string=tf.io.read_file(filename)

    image_decoded=tf.image.decode_png(image_string,channels=3)

    #image_decoded=image_aug(image_decoded)

    image_resized=tf.image.resize(image_decoded,[IMG_WIDTH,IMG_HEIGHT])

    image_normalized=image_resized/255.0

    label=tf.dtypes.cast(label,tf.int32)

    label=tf.one_hot(label,5)

    return image_normalized,label







def parse_function_for_validate(filename,label):

    image_string=tf.io.read_file(filename)

    image_decoded=tf.image.decode_png(image_string,channels=3)

    image_resized=tf.image.resize(image_decoded,[IMG_WIDTH,IMG_HEIGHT])

    image_normalized=image_resized/255.0

    label=tf.dtypes.cast(label,tf.int32)

    label=tf.one_hot(label,5)

    return image_normalized,label







def image_aug(img):

    img=tf.image.adjust_gamma(img,gamma=1, gain=1)

    img=tf.image.adjust_contrast(img,1)

    #img = tf.image.random_flip_left_right(img) horizontal flip

    img=X = tf.image.random_flip_up_down(img) #vertical flip

    img = tf.image.random_brightness(img, max_delta = 0.1)

    img = tf.image.random_saturation(img, lower = 0.75, upper = 1.5)

    img = tf.image.random_hue(img, max_delta = 0.15)

    img = tf.image.random_contrast(img, lower = 0.75, upper = 1.5)

    return img



def create_dataset(filenames, labels, is_training=True):

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if is_training:

        dataset = dataset.map(parse_function_for_train, num_parallel_calls=AUTOTUNE)

    else:

        dataset = dataset.map(parse_function_for_validate, num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset
train_dataset=create_dataset(train_files_path,train_labels)
validation_dataset=create_dataset(validation_files_path,validation_labels,is_training=False)




def print_image_from_dataset(dataset,number):

    images_ds=dataset.map(lambda image,label :image).unbatch()

    labels_ds=dataset.map(lambda image,label :label).unbatch()

    images=next(iter(images_ds.batch(validation_df.shape[0]))).numpy()

    labels=next(iter(labels_ds.batch(validation_df.shape[0]))).numpy()

    for i in range(number):

        print(images[i].shape)

        plt.imshow(images[i])

        plt.title(labels[i])

        plt.show()



print_image_from_dataset(train_dataset,10) #we are checking if the augmentation worked properly
!pip install -q efficientnet

import efficientnet.tfkeras as efn
metrices=[tf.keras.metrics.CategoricalAccuracy(name='acc')]
with tpu_strategy.scope():

    enet = efn.EfficientNetB7(

        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),

        weights='imagenet',#'imagenet if training for first time'

        include_top=False

    )

    enet.trainable = True

    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(5, activation='softmax')

    ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=metrices

    )

    model.summary()
model.load_weights('/kaggle/input/aptos-blindness-detection-augmentation/efficientnetb7epochs36weightswithoversamplingnoimgaugpart0.hdf5')
histories=[]

for i in range(EPOCHS):

    print("EPOCHS",10+i+1)

    history = model.fit(

        train_dataset, 

        epochs=1,

        steps_per_epoch=STEPS_PER_EPOCH,

        validation_data=validation_dataset,validation_steps=2

    )

    histories.append(history)
model.evaluate(validation_dataset)
model.save_weights('efficientnetb7epochs20weightswithoversamplingnoimgaugpart1.hdf5')
historiestoarray=[]

for x in histories:

    historiestoarray.append(x.history)
from numpy import save

save("history1to10.npy",historiestoarray)