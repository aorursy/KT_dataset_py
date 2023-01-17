import re

import os

import numpy as np

import pandas as pd

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



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

IMAGE_SIZE = [180, 180]

EPOCHS = 25
filenames = tf.io.gfile.glob(str(GCS_PATH + '/chest_xray/train/*/*'))

filenames.extend(tf.io.gfile.glob(str(GCS_PATH + '/chest_xray/val/*/*')))



train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])

print("Normal images count in training set: " + str(COUNT_NORMAL))



COUNT_PNEUMONIA = len([filename for filename in train_filenames if "PNEUMONIA" in filename])

print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))
train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)

val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)



for f in train_list_ds.take(5):

    print(f.numpy())
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()

print("Training images count: " + str(TRAIN_IMG_COUNT))



VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()

print("Validating images count: " + str(VAL_IMG_COUNT))
CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]

                        for item in tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*"))])

CLASS_NAMES
def get_label(file_path):

    # convert the path to a list of path components

    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory

    return parts[-2] == "PNEUMONIA"
def decode_img(img):

  # convert the compressed string to a 3D uint8 tensor

  img = tf.image.decode_jpeg(img, channels=3)

  # Use `convert_image_dtype` to convert to floats in the [0,1] range.

  img = tf.image.convert_image_dtype(img, tf.float32)

  # resize the image to the desired size.

  return tf.image.resize(img, IMAGE_SIZE)
def process_path(file_path):

    label = get_label(file_path)

    # load the raw data from the file as a string

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    return img, label
train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)



val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in train_ds.take(1):

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())
test_list_ds = tf.data.Dataset.list_files(str(GCS_PATH + '/chest_xray/test/*/*'))

TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()

test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE)



TEST_IMAGE_COUNT
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):

    # This is a small dataset, only load it once, and keep it in memory.

    # use `.cache(filename)` to cache preprocessing work for datasets that don't

    # fit in memory.

    if cache:

        if isinstance(cache, str):

            ds = ds.cache(cache)

        else:

            ds = ds.cache()



    ds = ds.shuffle(buffer_size=shuffle_buffer_size)



    # Repeat forever

    ds = ds.repeat()



    ds = ds.batch(BATCH_SIZE)



    # `prefetch` lets the dataset fetch batches in the background while the model

    # is training.

    ds = ds.prefetch(buffer_size=AUTOTUNE)



    return ds
train_ds = prepare_for_training(train_ds)

val_ds = prepare_for_training(val_ds)



image_batch, label_batch = next(iter(train_ds))
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        if label_batch[n]:

            plt.title("PNEUMONIA")

        else:

            plt.title("NORMAL")

        plt.axis("off")
show_batch(image_batch.numpy(), label_batch.numpy())
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

        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        

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

        

        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    

    return model
initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])

initial_bias
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 

weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
with strategy.scope():

    model = build_model()



    METRICS = [

        'accuracy',

        tf.keras.metrics.Precision(name='precision'),

        tf.keras.metrics.Recall(name='recall')

    ]

    

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=METRICS

    )
history = model.fit(

    train_ds,

    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=val_ds,

    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,

    class_weight=class_weight,

)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5",

                                                    save_best_only=True)



early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,

                                                     restore_best_weights=True)
def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1 **(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(0.01, 20)



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(

    train_ds,

    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,

    epochs=100,

    validation_data=val_ds,

    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,

    class_weight=class_weight,

    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]

)
fig, ax = plt.subplots(1, 4, figsize=(20, 3))

ax = ax.ravel()



for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
loss, acc, prec, rec = model.evaluate(test_ds)