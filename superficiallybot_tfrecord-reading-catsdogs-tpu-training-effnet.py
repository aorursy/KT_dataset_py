!pip install -q efficientnet



import efficientnet.tfkeras as efn
import re

import math

import numpy as np

import seaborn as sns
from kaggle_datasets import KaggleDatasets

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



import tensorflow as tf

from tensorflow.keras import layers
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

    

print('REPLICAS : -> ', strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('catsdogstfrecords192x192')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/train.*tfrecords')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/val.*tfrecords')

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [192,192]

print(TRAINING_FILENAMES)

print(VALIDATION_FILENAMES)

print(BATCH_SIZE)

print(IMAGE_SIZE)
EPOCHS = 10

STEPS_PER_EPOCH = 15000 // BATCH_SIZE
def decode_image(image_data):

    print('About to decode image data...')

    #image = tf.image.decode_jpeg(image_data, channels = 3)

    #image = tf.io.parse_tensor(image_data, out_type = tf.uint8)

    image = tf.io.decode_raw(image_data, tf.uint8)

    print('Decoded JPEG...')

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    print('Done decoding image')

    return image
def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image_raw" : tf.io.FixedLenFeature([], tf.string),

        "label" : tf.io.FixedLenFeature([], tf.int64)

    }

    

    print('About to parse labeled tfrecord...')

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image_raw'])

    print('Read Image Data...')

    label = tf.cast(example['label'], tf.int32)

    return image, label
def load_dataset(filenames, labeled = True, ordered = False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False

    

    print('About to Load TFRECORD Dataset...')

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = tf.data.experimental.AUTOTUNE)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    print('Read labeled TFRecords...')

    return dataset
def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled = True)

    print('loaded training dataset')

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print('Successfully loaded training dataset')

    return dataset

    
def get_validation_dataset():

    dataset = load_dataset(VALIDATION_FILENAMES, labeled = True)

    print('loaded training dataset')

    dataset = dataset.repeat(1)

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print('Successfully loaded validation dataset')

    return dataset

    
histories = []
with strategy.scope():

    enet = efn.EfficientNetB1(

        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

        weights = 'imagenet',

        include_top = False

    )

    

    enet.trainable = False

    model = tf.keras.Sequential([

        enet, 

        tf.keras.layers.GlobalMaxPooling2D(name = 'layer1'),

        tf.keras.layers.Dropout(0.4),

       # tf.keras.layers.Dense(1, activation = 'softmax')

        tf.keras.layers.Dense(1, activation = 'sigmoid')

    ])
"""# METRICS



model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001),

             loss = 'sparse_categorical_crossentropy',

              metrics= ['categorical_accuracy']

             )"""
"""# METRICS



model.compile(optimizer = 'tf.keras.optimizers.Adam(lr = 0.0001)',

             loss = 'categorical_crossentropy',

              metrics= ['accuracy']

             )"""
# METRICS



model.compile(optimizer = 'rmsprop',

             loss = 'binary_crossentropy',

              metrics= ['accuracy']

             )
training_history_effnet = model.fit(get_training_dataset(), 

         steps_per_epoch = STEPS_PER_EPOCH, 

         validation_data = get_validation_dataset(),

          epochs = EPOCHS

         )



histories.append(training_history_effnet)
import seaborn as sns; import matplotlib.pyplot as plt; sns.set()
histories[0].history
sns.lineplot(x = list(range(1,11)), y = histories[0].history['loss'], label = 'Training Loss', color = 'red');

sns.lineplot(x = list(range(1,11)), y = histories[0].history['val_loss'], label = 'Validation Loss', color = 'black').set_title('Loss Plot vs Epoch');

plt.show()
sns.lineplot(x = list(range(1,11)), y = histories[0].history['accuracy'], label = 'Training Accuracy', color = 'black');

sns.lineplot(x = list(range(1,11)), y = histories[0].history['val_accuracy'], label = 'Validation Accuracy', color = 'red').set_title('Accuracy Plot vs Epoch');

plt.show()