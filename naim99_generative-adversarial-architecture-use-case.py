import tensorflow as tf

import numpy as np

import datetime

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import tensorflow_addons as tfa



from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import numpy as np



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)



AUTOTUNE = tf.data.experimental.AUTOTUNE

    

print(tf.__version__)

GCS_PATH = KaggleDatasets().get_gcs_path()

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))

print('Monet TFRecord Files:', len(MONET_FILENAMES))



PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))

print('Photo TFRecord Files:', len(PHOTO_FILENAMES))