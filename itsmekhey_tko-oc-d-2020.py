!pip install -q efficientnet
import numpy as np

import pandas as pd

import os

import random, re, math, time

random.seed(a=128)



from os.path import join 



import tensorflow as tf

import tensorflow.keras.backend as K

#import tensorflow_addons as tfa

import efficientnet.tfkeras as efn



from tqdm.keras import TqdmCallback



from PIL import Image

import PIL



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold



from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

import plotly

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



from pandas_summary import DataFrameSummary



from kaggle_datasets import KaggleDatasets



from tqdm import tqdm
DEVICE ="TPU"
if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    

print("REPLICAS: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE

# Configuration

EPOCHS = 20

BATCH_SIZE = 16* strategy.num_replicas_in_sync

IMAGE_SIZE = [512,512]
GCS_PATH= KaggleDatasets().get_gcs_path('oc-d-512512')

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

VALIDATION_FILENAMES =tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

TEST_FILENAMES =tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

TRAINING_FILENAMES = TRAINING_FILENAMES[:4]

VALIDATION_FILENAMES =VALIDATION_FILENAMES[4:] 

#TEST_FILENAMES =TEST_FILENAMES[4:]

CLASSES=[0,1,2,3,4,5,6,7]
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "filename": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['filename']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_hue(image, 0.01)

    image = tf.image.random_saturation(image, 0.7, 1.3)

    image = tf.image.random_contrast(image, 0.8, 1.2)

    image = tf.image.random_brightness(image, 0.1)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

NUM_VALIDATION_IMAGES= count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
def lrfn(epoch):

    LR_START          = 0.000005

    LR_MAX            = 0.000020 * strategy.num_replicas_in_sync

    LR_MIN            = 0.000001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr





from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate,multiply, LocallyConnected2D, Lambda)

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

import keras

from keras.models import Model

CFG = dict(

    inp_size          = 512,

    read_size         = 512, 

    crop_size         = 512,

    net_size          = 512,

   

)

!pip install -q efficientnet

import efficientnet.tfkeras as efn



from keras.activations import hard_sigmoid
with strategy.scope():

    in_lay = Input(shape=(CFG['inp_size'], CFG['inp_size'],3))

    base_model = efn.EfficientNetB7(weights='noisy-student',

        input_shape=(CFG['inp_size'], CFG['inp_size'],3),

        include_top=False

                       )

    #base_model.load_weights("../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5")

    pt_depth = base_model.get_output_shape_at(0)[-1]

    pt_features = base_model(in_lay)

    bn_features = BatchNormalization()(pt_features)

    

    # ici nous faisons un mécanisme d'attention pour activer et désactiver les pixels dans le GAP

    # lidee est baser sur cette explication 

    #1-http://akosiorek.github.io/ml/2017/10/14/visual-attention.html

    #2-https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/

    

    

    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.25)(bn_features))

    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

    attn_layer = Conv2D(1, 

                        kernel_size = (1,1), 

                        padding = 'valid', 

                        activation = 'sigmoid')(attn_layer)

    # diffusez sur toutes les chaînes

    # kernel_size  détermine les dimensions du noyau. Les dimensions courantes comprennent 1×1, 3×3, 5×5 et 7×7, qui peuvent être passées en (1, 1), (3, 3), (5, 5) ou (7, 7) tuples.

    # Il s'agit d'un nombre entier ou d'un tuple/liste de 2 nombres entiers, spécifiant la hauteur et la largeur de la fenêtre de convolution 2D.

    #  Ce paramètre doit être un nombre entier impair

    # pour plus de details sur cette partie (mask et use_bias ... ) il ya  une bonne explication sur geekforgeeks

    #https://www.geeksforgeeks.org/keras-conv2d-class/

    

    up_c2_w = np.ones((1, 1, 1, pt_depth))

    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 

                   activation = 'linear', use_bias = False, weights = [up_c2_w])

    up_c2.trainable = False

    attn_layer = up_c2(attn_layer)



    mask_features = multiply([attn_layer, bn_features])

    gap_features = GlobalAveragePooling2D()(mask_features)

    gap_mask = GlobalAveragePooling2D()(attn_layer)

    

    # pour tenir compte des valeurs manquantes du modèle d'attention

    # pour bien comprendre resaclegap il ya un bon exemple ici qui explique tellemnt bien cette partie 

    # https://codefellows.github.io/sea-python-401d5/lectures/rescaling_data.html

    

    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])

    gap_dr = Dropout(0.25)(gap)

    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))

    out_layer = Dense(8, activation = 'softmax')(dr_steps)

    model = Model(inputs = [in_lay], outputs = [out_layer])

    model.summary()

    



from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
valid_ds = get_validation_dataset()



valid_images_ds = valid_ds.map(lambda image, label: image)

valid_labels_ds = valid_ds.map(lambda image, label: label).unbatch()



valid_labels = next(iter(valid_labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch



valid_steps = NUM_VALIDATION_IMAGES // BATCH_SIZE



if NUM_VALIDATION_IMAGES % BATCH_SIZE > 0:

    valid_steps += 1
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

"""

for layer in model.layers[:-8]:

    layer.trainable=False

    

for layer in model.layers[-8:]:

    layer.trainable=True    

"""



model.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



history = model.fit(

    get_training_dataset(), 

    steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[lr_schedule],

    validation_data=valid_ds

)


