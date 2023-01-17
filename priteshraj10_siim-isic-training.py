!pip install -q efficientnet
import numpy as np

import pandas as pd

import os

import math

import re

from matplotlib import pyplot as plt

from sklearn import metrics

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

DEBUG = False

N_FOLD = 4

EPOCHS = 1 if DEBUG else 7

BATCH_SIZE = 8* strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')

TRAINING_FILENAMES_LIST = [TRAINING_FILENAMES[:4], TRAINING_FILENAMES[4:8], TRAINING_FILENAMES[8:12], TRAINING_FILENAMES[12:]]
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    return image, label



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    return dataset



def data_augment(image, label):

    image = tf.image.random_flip_left_right(image)

    return image, label



def get_training_dataset(train_files):

    dataset = load_dataset(train_files, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset(valid_files, ordered=False):

    dataset = load_dataset(valid_files, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset(test_files, ordered=False):

    dataset = load_dataset(test_files, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
def get_model():

    with strategy.scope():

        model = tf.keras.Sequential([

            efn.EfficientNetB3(

            input_shape = (*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

            ),

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

        model.compile(

            optimizer='adam', 

            loss='binary_crossentropy',

            metrics = [tf.keras.metrics.AUC()],

        )

        return model
def train_model(fold, debug=False):

    if debug:

        valid = TRAINING_FILENAMES_LIST[fold][0:1]

        train = TRAINING_FILENAMES_LIST[fold-1][0:1]

    else:

        valid = TRAINING_FILENAMES_LIST[fold]

        train = sum([TRAINING_FILENAMES_LIST[i] for i in range(N_FOLD) if i not in [fold]], [])

    num_train = count_data_items(train)

    steps_per_epoch = num_train // BATCH_SIZE

    model = get_model()

    saving_callback = tf.keras.callbacks.ModelCheckpoint(f"fold{fold}_model.h5", verbose=1, save_weights_only=True, save_best_only=True)

    history = model.fit(

        get_training_dataset(train),

        steps_per_epoch=steps_per_epoch,

        epochs=EPOCHS,

        callbacks=[saving_callback],

        validation_data=get_validation_dataset(valid),

        verbose=1

    )
for fold in range(N_FOLD):

    train_model(fold, debug=DEBUG)
def get_model():

    with strategy.scope():

        model = tf.keras.Sequential([

            efn.EfficientNetB3(

                input_shape=(*IMAGE_SIZE, 3),

                weights=None,

                include_top=False

            ),

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

    

    return model
from tqdm import tqdm



oof_df = pd.DataFrame()



tk0 = tqdm(range(N_FOLD), total=N_FOLD)



for fold in tk0:

    if DEBUG:

        test_files = TRAINING_FILENAMES_LIST[fold-1][0:1]

    else:

        test_files = TRAINING_FILENAMES_LIST[fold]

    num_test = count_data_items(test_files)

    test_ds = get_test_dataset(test_files, ordered=True)

    test_images_ds = test_ds.map(lambda image, idnum: image)

    model = get_model()

    model.load_weights(f"fold{fold}_model.h5")

    probabilities = model.predict(test_images_ds)

    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

    test_ids = next(iter(test_ids_ds.batch(num_test))).numpy().astype('U')

    _oof_df = pd.DataFrame({'image_name': test_ids, 'oof': np.concatenate(probabilities)})

    oof_df = pd.concat([oof_df, _oof_df])
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

train_df = train_df.merge(oof_df, on='image_name')

train_df.to_csv('oof_df.csv', index=False)
from sklearn.metrics import roc_auc_score

score = roc_auc_score(train_df['target'].values, train_df['oof'].values)

print(f'CV AUC:{score}')