!pip install -q efficientnet
import math, re, os



import numpy as np

import pandas as pd

from kaggle_datasets import KaggleDatasets

import tensorflow as tf



import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn



from sklearn.model_selection import train_test_split

import cv2

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path("alaska2-image-steganalysis")



# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IM_SIZE = 512
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')

train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))
# np.random.seed(0)

positives = train_filenames.copy()

negatives = train_filenames.copy()

np.random.shuffle(positives)

np.random.shuffle(negatives)



jmipod = append_path('JMiPOD')(positives[:10000])

juniward = append_path('JUNIWARD')(positives[10000:20000])

uerd = append_path('UERD')(positives[20000:30000])



pos_paths = np.concatenate([jmipod, juniward, uerd])
test_paths = append_path('Test')(sub.Id.values)

neg_paths = append_path('Cover')(negatives[:30000])
train_paths = np.concatenate([pos_paths, neg_paths])

train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
train_paths, valid_paths, train_labels, valid_labels = train_test_split(

    train_paths, train_labels, test_size=0.15)
def decode_image(filename, label=None, image_size=(IM_SIZE, IM_SIZE)):

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

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .shuffle(1024)

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
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE



checkpoint = tf.keras.callbacks.ModelCheckpoint('model_b3.h5',monitor = 'val_loss', save_best_only = True, verbose = 1, period = 1)



reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', min_lr = 0.00001, patience = 3, mode = 'min', verbose = 1)
with strategy.scope():

    model_efn = tf.keras.models.load_model('../input/alaska-tpu/model_b3.h5')
history = model_efn.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = valid_dataset,

                 callbacks = [checkpoint, reduceLR])
with strategy.scope():

    model_res = tf.keras.models.load_model('../input/alaska-tpu/model_b7.h5')
history = model_res.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = valid_dataset,

                 callbacks = [checkpoint, reduceLR])
probs = (model_res.predict(test_dataset, verbose=1)\

         +model_efn.predict(test_dataset, verbose=1))/2
sub.Label = probs

sub.to_csv('submission.csv', index=False)

sub.head()