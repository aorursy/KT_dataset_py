!pip install -q efficientnet

import numpy as np 

import pandas as pd 

import re

import cv2

import math

import time

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Concatenate, Flatten, Dense

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn

import os
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE



try: ## Trying to check if a TPU cluster exists

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('TPU Master ', tpu.master)

except ValueError:

    tpu = None

    

if tpu: #In the case the cluster exists, we initialize it and connect to it and create a strategy for parallel processing

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

    

print('Replicas', strategy.num_replicas_in_sync)    



DATASET1 = 'melanoma-768x768'

GCS_PATH1 = KaggleDatasets().get_gcs_path(DATASET1) #Getting the Google Cloud Storage path for the publically available dataset
train_details = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

train_details.isna().sum()
train_details['sex'] = train_details['sex'].fillna('male')

train_details['age_approx'] = train_details['age_approx'].fillna(train_details['age_approx'].mean())

train_details['anatom_site_general_challenge'] = train_details['anatom_site_general_challenge'].fillna('head/neck')
from sklearn.preprocessing import LabelEncoder

enc1 = LabelEncoder()

enc2 = LabelEncoder()



train_details['sex'] = enc1.fit_transform(train_details['sex'])

train_details['anatom_site_general_challenge'] = enc2.fit_transform(train_details['anatom_site_general_challenge'])



x_vec = train_details[['sex','age_approx','anatom_site_general_challenge']]
SEED = 42

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

SIZE1 = [768,768]

LR = 0.00004

EPOCHS = 12

WARMUP = 5

WEIGHT_DECAY = 0

LABEL_SMOOTHING = 0.05

TTA = 4
np.random.seed(SEED)

tf.random.set_seed(SEED)
train_filenames1 = tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec')

test_filenames1 = tf.io.gfile.glob(GCS_PATH1 + '/test*.tfrec')
from sklearn.model_selection import train_test_split

train_filenames1, valid_filenames1 = train_test_split(train_filenames1, test_size = 0.15, random_state = SEED)
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels = 3)

    image = tf.cast(image, tf.float32)/255.0

    image = tf.reshape(image, [*SIZE1, 3])

    return image



def data_augment(image, label = None, seed = SEED):

    image = tf.image.rot90(image, k = np.random.randint(4))

    image = tf.image.random_flip_left_right(image, seed = SEED)

    image = tf.image.random_flip_up_down(image, seed = SEED)

    if label is None:

        return image

    else:

        return image, label



def read_labeled_tfrecord(example):

    LFormat = {'image': tf.io.FixedLenFeature([], tf.string),

             'target': tf.io.FixedLenFeature([], tf.int64)}

    example = tf.io.parse_single_example(example, LFormat)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    

    return image, label



def read_unlabeled_tfrecord(example):

    UFormat = {'image': tf.io.FixedLenFeature([], tf.string),

             'image_name': tf.io.FixedLenFeature([], tf.string)}

    example = tf.io.parse_single_example(example, UFormat)

    image = decode_image(example['image'])

    image_name = example['image_name']

    

    return image, image_name



def load_dataset(filenames, labeled = True, ordered = False):

    ignore_order = tf.data.Options()

    

    if not ordered:

        ignore_order.experimental_deterministic = False

    

    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO).with_options(ignore_order).map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO))

    

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
train1 = (load_dataset(train_filenames1).map(data_augment, num_parallel_calls = AUTO).shuffle(SEED).batch(BATCH_SIZE, drop_remainder = True).repeat().prefetch(AUTO))

valid_dataset1 = (load_dataset(valid_filenames1, labeled=True).batch(BATCH_SIZE).repeat().prefetch(AUTO))
with strategy.scope():

    model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(*SIZE1,3), weights='imagenet', include_top=False, pooling = 'avg'),

            Dense(1, activation = 'sigmoid')])

    

    model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),

        metrics=[tf.keras.metrics.AUC(name='auc')])
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):

    def lrfn(epoch):

        if epoch < num_warmup_steps:

            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr



    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)
validationsize1 = count_data_items(valid_filenames1)

print(validationsize1)
STEPS_PER_EPOCH = (count_data_items(train_filenames1)) // BATCH_SIZE

#class_weights = {0:0.5089, 1:28.3613}

model.fit(train1, epochs=EPOCHS, callbacks=[lr_schedule],steps_per_epoch=STEPS_PER_EPOCH,validation_data=valid_dataset1,validation_steps=validationsize1//BATCH_SIZE)
num_test_images = count_data_items(test_filenames1)

submission_df1 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

for i in range(TTA):

    test_dataset = (load_dataset(test_filenames1, labeled=False,ordered=True)

    .map(data_augment, num_parallel_calls=AUTO)  

    .batch(BATCH_SIZE))

    test_dataset_images = test_dataset.map(lambda image, image_name: image)

    test_dataset_image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()

    test_ids = next(iter(test_dataset_image_name.batch(num_test_images))).numpy().astype('U')

    test_pred = model.predict(test_dataset_images, verbose=1) 

    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(test_pred)})

    temp = submission_df1.copy()   

    del temp['target']  

    submission_df1['target'] += temp.merge(pred_df,on="image_name")['target']/TTA
submission_df1.to_csv('efficientnetb7_784.csv', index = False)
DATASET2 = 'melanoma-512x512'

GCS_PATH2 = KaggleDatasets().get_gcs_path(DATASET2) #Getting the Google Cloud Storage path for the publically available dataset



SIZE2 = [512,512]



train_filenames2 = tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec')

test_filenames2 = tf.io.gfile.glob(GCS_PATH2 + '/test*.tfrec')



train_filenames2, valid_filenames2 = train_test_split(train_filenames2, test_size = 0.15, random_state = SEED)



def decode_image(image):

    image = tf.image.decode_jpeg(image, channels = 3)

    image = tf.cast(image, tf.float32)/255.0

    image = tf.reshape(image, [*SIZE2, 3])

    return image



def data_augment(image, label = None, seed = SEED):

    image = tf.image.rot90(image, k = np.random.randint(4))

    image = tf.image.random_flip_left_right(image, seed = SEED)

    image = tf.image.random_flip_up_down(image, seed = SEED)

    if label is None:

        return image

    else:

        return image, label



def read_labeled_tfrecord(example):

    LFormat = {'image': tf.io.FixedLenFeature([], tf.string),

             'target': tf.io.FixedLenFeature([], tf.int64)}

    example = tf.io.parse_single_example(example, LFormat)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    

    return image, label



def read_unlabeled_tfrecord(example):

    UFormat = {'image': tf.io.FixedLenFeature([], tf.string),

             'image_name': tf.io.FixedLenFeature([], tf.string)}

    example = tf.io.parse_single_example(example, UFormat)

    image = decode_image(example['image'])

    image_name = example['image_name']

    

    return image, image_name



def load_dataset(filenames, labeled = True, ordered = False):

    ignore_order = tf.data.Options()

    

    if not ordered:

        ignore_order.experimental_deterministic = False

    

    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO).with_options(ignore_order).map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO))

    

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



train2 = (load_dataset(train_filenames2).map(data_augment, num_parallel_calls = AUTO).shuffle(SEED).batch(BATCH_SIZE, drop_remainder = True).repeat().prefetch(AUTO))

valid_dataset2 = (load_dataset(valid_filenames2, labeled=True).batch(BATCH_SIZE).repeat().prefetch(AUTO))
with strategy.scope():



    model2 = tf.keras.Sequential([

        efn.EfficientNetB7(input_shape=(*SIZE2, 3),weights='imagenet',pooling='avg',include_top=False),

        Dense(1, activation='sigmoid')

    ])

    

    model2.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),

        metrics=[tf.keras.metrics.AUC(name='auc')])
validationsize2 = count_data_items(valid_filenames2)

print(validationsize2)
STEPS_PER_EPOCH = (count_data_items(train_filenames2)) // BATCH_SIZE

#class_weights = {0:0.5089, 1:28.3613}

model2.fit(train2, epochs=EPOCHS, callbacks=[lr_schedule],steps_per_epoch=STEPS_PER_EPOCH,validation_data=valid_dataset2,validation_steps=validationsize2//BATCH_SIZE)
num_test_images = count_data_items(test_filenames2)

submission_df2 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

for i in range(TTA):

    test_dataset = (load_dataset(test_filenames2, labeled=False,ordered=True)

    .map(data_augment, num_parallel_calls=AUTO)  

    .batch(BATCH_SIZE))

    test_dataset_images = test_dataset.map(lambda image, image_name: image)

    test_dataset_image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()

    test_ids = next(iter(test_dataset_image_name.batch(num_test_images))).numpy().astype('U')

    test_pred = model2.predict(test_dataset_images, verbose=1) 

    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(test_pred)})

    temp = submission_df2.copy()   

    del temp['target']  

    submission_df2['target'] += temp.merge(pred_df,on="image_name")['target']/TTA
submission_df2.to_csv('efficientnetb7_512.csv', index=False)