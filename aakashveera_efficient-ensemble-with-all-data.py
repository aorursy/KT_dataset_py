import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import os

import random

import re

import math

import time

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')
!pip install -q efficientnet
import tensorflow as tf

import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets
train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

sample = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")
train.head()
GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')

GCS_PATH2 = KaggleDatasets().get_gcs_path('malignant-v2-384x384')

GCS_PATH3 = KaggleDatasets().get_gcs_path('isic2019-384x384')

filenames_train1 = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

filenames_train2 = tf.io.gfile.glob(GCS_PATH2 + '/train%.2i*.tfrec'%(2*x) for x in range(15))

filenames_train3 = tf.io.gfile.glob(GCS_PATH3 + '/train%.2i*.tfrec'%(2*x) for x in range(15))

filenames_test = np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec'))
filenames_train = np.array(filenames_train1+filenames_train2+filenames_train3)

np.random.shuffle(filenames_train)
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

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
cfg = dict(

           batch_size=32,

           img_size=384,

    

           lr_start=0.000005,

           lr_max=0.00000125,

           lr_min=0.000001,

           lr_rampup=5,

           lr_sustain=0,

           lr_decay=0.8,

           epochs=12,

    

           transform_prob=1.0,

           rot=180.0,

           shr=2.0,

           hzoom=8.0,

           wzoom=8.0,

           hshift=8.0,

           wshift=8.0,

    

           optimizer='adam',

           label_smooth_fac=0.05,

           tta_steps=20

            

        )
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):



    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.



    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1], dtype='float32')

    zero = tf.constant([0], dtype='float32')

    rotation_matrix = tf.reshape(

        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),

        [3, 3])



    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape(

        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),

        [3, 3])



    # ZOOM MATRIX

    zoom_matrix = tf.reshape(

        tf.concat([

            one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero,

            zero, one

        ],axis=0), [3, 3])



    # SHIFT MATRIX

    shift_matrix = tf.reshape(

        tf.concat(

            [one, zero, height_shift, zero, one, width_shift, zero, zero, one],

            axis=0), [3, 3])



    return K.dot(K.dot(rotation_matrix, shear_matrix),

                 K.dot(zoom_matrix, shift_matrix))

def transform(image, cfg):

    

    DIM = cfg['img_size']

    XDIM = DIM % 2  # fix for size 331



    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')

    shr = cfg['shr'] * tf.random.normal([1], dtype='float32')

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']

    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32')

    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32')



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)

    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])

    z = tf.ones([DIM * DIM], dtype='int32')

    idx = tf.stack([x, y, z])



    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)



    # FIND ORIGIN PIXEL VALUES

    idx3 = tf.stack([DIM // 2 - idx2[0, ], DIM // 2 - 1 + idx2[1, ]])

    d = tf.gather_nd(image, tf.transpose(idx3))



    return tf.reshape(d, [DIM, DIM, 3])
def dropout(image, DIM=384, PROBABILITY = 0.75, CT = 8, SZ = 0.2):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image with CT squares of side size SZ*DIM removed

    

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE

    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)

    if (P==0)|(CT==0)|(SZ==0): return image

    

    for k in range(CT):

        # CHOOSE RANDOM LOCATION

        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        # COMPUTE SQUARE 

        WIDTH = tf.cast( SZ*DIM,tf.int32) * P

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        # DROPOUT IMAGE

        one = image[ya:yb,0:xa,:]

        two = tf.zeros([yb-ya,xb-xa,3]) 

        three = image[ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 

    image = tf.reshape(image,[DIM,DIM,3])

    return image
def prepare_image(img, cfg=None,droprate=0.5,dropct=8,dropsize=0.2):

    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [cfg['img_size'], cfg['img_size']],

                          antialias=True)

    img = tf.cast(img, tf.float32) / 255.0



    if cfg['transform_prob'] > tf.random.uniform([1], minval=0, maxval=1):

        img = transform(img, cfg)

    

    if (tf.random.uniform([1], minval=0, maxval=1) > 0.5) & (droprate!=0)&(dropct!=0)&(dropsize!=0):

        img = dropout(img, DIM=384, PROBABILITY=droprate, CT=dropct, SZ=dropsize)



    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_saturation(img, 0.7, 1.3)

    img = tf.image.random_contrast(img, 0.8, 1.2)

    img = tf.image.random_brightness(img, 0.1)



    return img
def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'image_name': tf.io.FixedLenFeature([], tf.string),

        'target': tf.io.FixedLenFeature([], tf.int64)

    }



    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    return example['image'], example['target']

def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'image_name': tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    return example['image'], example['image_name']



def count_data_items(filenames):

    n = [

        int(re.compile(r'-([0-9]*)\.').search(filename).group(1))

        for filename in filenames

    ]

    return np.sum(n)
def getTrainDataset(files, cfg):

    

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()



    opt = tf.data.Options()

    opt.experimental_deterministic = False

    ds = ds.with_options(opt)



    ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    ds = ds.repeat()

    

    ds = ds.shuffle(2048)

    ds = ds.map(lambda img, label:

                (prepare_image(img, cfg=cfg), label),

                num_parallel_calls=AUTO)

    ds = ds.batch(cfg['batch_size'] * strategy.num_replicas_in_sync)

    ds = ds.prefetch(AUTO)

    return ds



def getTestDataset(files, cfg, augment=False, repeat=False):

    

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    if repeat:

        ds = ds.repeat()

    ds = ds.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    ds = ds.map(lambda img, idnum:

                (prepare_image(img, cfg=cfg), idnum),

                num_parallel_calls=AUTO)

    ds = ds.batch(cfg['batch_size'] * strategy.num_replicas_in_sync)

    ds = ds.prefetch(AUTO)

    return ds
def getLearnRateCallback(cfg):

    

    lr_start = cfg['lr_start']

    lr_max = cfg['lr_max'] * strategy.num_replicas_in_sync * cfg['batch_size']

    lr_min = cfg['lr_min']

    lr_rampup = cfg['lr_rampup']

    lr_sustain = cfg['lr_sustain']

    lr_decay = cfg['lr_decay']



    def lrfn(epoch):

        if epoch < lr_rampup:

            lr = (lr_max - lr_start) / lr_rampup * epoch + lr_start

        elif epoch < lr_rampup + lr_sustain:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_rampup -

                                                lr_sustain) + lr_min

        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback
with strategy.scope():

    model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),

                                 name='img_input')



    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)



    outputs = []



    x = efn.EfficientNetB3(include_top=False,

                           weights='noisy-student',

                           input_shape=(cfg['img_size'], cfg['img_size'], 3),

                           pooling='avg')(dummy)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    outputs.append(x)



    x = efn.EfficientNetB4(include_top=False,

                           weights='noisy-student',

                           input_shape=(cfg['img_size'], cfg['img_size'], 3),

                           pooling='avg')(dummy)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    outputs.append(x)



    x = efn.EfficientNetB5(include_top=False,

                           weights='noisy-student',

                           input_shape=(cfg['img_size'], cfg['img_size'], 3),

                           pooling='avg')(dummy)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    outputs.append(x)

    

    model = tf.keras.Model(model_input, outputs, name='aNetwork')



    model.compile(optimizer=cfg['optimizer'],

                  loss=[

                      tf.keras.losses.BinaryCrossentropy(

                          label_smoothing=cfg['label_smooth_fac']),

                      tf.keras.losses.BinaryCrossentropy(

                          label_smoothing=cfg['label_smooth_fac']),

                      tf.keras.losses.BinaryCrossentropy(

                          label_smoothing=cfg['label_smooth_fac'])

                  ],

                  metrics=[tf.keras.metrics.AUC(name='auc')])

model.summary()
ds_train = getTrainDataset(filenames_train, cfg).map(lambda img, label: (img, (label, label, label)))



stepsTrain = count_data_items(filenames_train) /(cfg['batch_size'] * strategy.num_replicas_in_sync)
callbacks = [getLearnRateCallback(cfg)]



history = model.fit(ds_train,

                    validation_data=None,

                    verbose=1,

                    steps_per_epoch=stepsTrain,

                    validation_steps=0,

                    epochs=10,

                    callbacks=callbacks)
steps = count_data_items(filenames_test) / (cfg['batch_size'] * strategy.num_replicas_in_sync)

z = np.zeros((cfg['batch_size'] * strategy.num_replicas_in_sync))



ds_testAug = getTestDataset(filenames_test, cfg, augment=True,

    repeat=True).map(lambda img, label: (img, (z, z, z)))



probs = model.predict(ds_testAug, verbose=1, steps=steps * cfg['tta_steps'])
probs = np.stack(probs)

probs = probs[:, :count_data_items(filenames_test) * cfg['tta_steps']]

probs = np.stack(np.split(probs, cfg['tta_steps'], axis=1), axis=1)

probs = np.mean(probs, axis=1)
y_test_sorted = np.zeros((3, probs.shape[1]))

test = test.reset_index()

test = test.set_index('image_name')



i = 0

ds_test = getTestDataset(filenames_test, cfg)

for img, imgid in tqdm(iter(ds_test.unbatch())):

    imgid = imgid.numpy().decode('utf-8')

    y_test_sorted[:, test.loc[imgid]['index']] = probs[:, i, 0]

    i += 1



for i in range(y_test_sorted.shape[0]):

    submission = sample

    submission['target'] = y_test_sorted[i]

    submission.to_csv('model_%s.csv' % i, index=False)



submission = sample

submission['target'] = np.mean(y_test_sorted, axis=0)

submission.to_csv('ensembled.csv', index=False)
!gzip model_0.csv

!gzip model_1.csv

!gzip model_2.csv

!gzip ensembled.csv