!pip install tensorflow==2.3.0 -q
import os

import numpy as np

import pandas as pd

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import PIL



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

    

print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [176, 208]

EPOCHS = 100
GCS_PATH
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train",

    validation_split=0.2,

    subset="training",

    seed=1337,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

)
class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

train_ds.class_names = class_names

val_ds.class_names = class_names



NUM_CLASSES = len(class_names)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

  for i in range(9):

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(train_ds.class_names[labels[i]])

    plt.axis("off")
def one_hot_label(image, label):

    label = tf.one_hot(label, NUM_CLASSES)

    return image, label



train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
NUM_IMAGES = []



for label in class_names:

    dir_name = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/" + label[:-2] + 'ed'

    NUM_IMAGES.append(len([name for name in os.listdir(dir_name)]))
NUM_IMAGES
def conv_block(filters):

    block = tf.keras.Sequential([

        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),

        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D()

    ]

    )

    

    return block
def dense_block(units, dropout_rate):

    block = tf.keras.Sequential([

        tf.keras.layers.Dense(units, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(dropout_rate)

    ])

    

    return block
def build_model():

    model = tf.keras.Sequential([

        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),

        

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),

        tf.keras.layers.MaxPool2D(),

        

        conv_block(32),

        conv_block(64),

        

        conv_block(128),

        tf.keras.layers.Dropout(0.2),

        

        conv_block(256),

        tf.keras.layers.Dropout(0.2),

        

        tf.keras.layers.Flatten(),

        dense_block(512, 0.7),

        dense_block(128, 0.5),

        dense_block(64, 0.3),

        

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

    ])

    

    return model
with strategy.scope():

    model = build_model()



    METRICS = [tf.keras.metrics.AUC(name='auc')]

    

    model.compile(

        optimizer='adam',

        loss=tf.losses.CategoricalCrossentropy(),

        metrics=METRICS

    )
def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1 **(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(0.01, 20)



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)



checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.h5",

                                                    save_best_only=True)



early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,

                                                     restore_best_weights=True)
history = model.fit(

    train_ds,

    validation_data=val_ds,

    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],

    epochs=EPOCHS

)
fig, ax = plt.subplots(1, 2, figsize=(20, 3))

ax = ax.ravel()



for i, met in enumerate(['auc', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

)



test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
_ = model.evaluate(test_ds)