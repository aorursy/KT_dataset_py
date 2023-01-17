import math, re, os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow.keras.layers as L

from sklearn import metrics

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 40

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IM_Z = 768
def format_path(st):

    return GCS_DS_PATH + '/images/' + st + '.jpg'
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train.head()
train_paths = train.image_id.apply(format_path).values

test_paths = test.image_id.apply(format_path).values



train_labels = train.loc[:, 'healthy':].values



train_paths, valid_paths, train_labels, valid_labels = train_test_split(

    train_paths, train_labels, test_size=0.1, random_state=2020)
def decode_image(filename, label=None, image_size=(IM_Z, IM_Z)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

#     image = tf.image.adjust_brightness(image, delta=0.2)

#     image = tf.image.adjust_contrast(image,2)

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)

test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



ch_p = ModelCheckpoint(filepath="model_ir.h5", monitor='val_loss', save_weights_only=True,

                                                 verbose=1)
with strategy.scope():

    model = tf.keras.Sequential([

        tf.keras.applications.InceptionResNetV2(

            input_shape=(IM_Z, IM_Z, 3),

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(train_labels.shape[1], activation='softmax')

    ])

        

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

#     model.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE 
history = model.fit(

    train_dataset, 

    epochs=EPOCHS, 

    callbacks=[lr_schedule, ch_p],

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset

)
# model.load_weights("model_ir.h5")

probs = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs

sub.to_csv('submission.csv', index=False)

sub.head()