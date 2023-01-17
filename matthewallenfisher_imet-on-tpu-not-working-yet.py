import math, re, os

import tensorflow as tf # deep learning

import pandas as pd

import numpy as np # linear algebra

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version ", tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
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

    print('No TPU detected, look to your right under the "Accelerator" tab and switch to "TPU v3-8"')

print('REPLICAS: ', strategy.num_replicas_in_sync)
# Get the Google Cloud mirror path for this Kaggle dataset

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

!gsutil ls $GCS_DS_PATH
CLASS_NAMES = pd.read_csv(GCS_DS_PATH+'/labels.csv')

train_labels = pd.read_csv('/kaggle/input/imet-2020-fgvc7/train.csv')

ids = train_labels.pop('id')

attributes = train_labels.pop('attribute_ids')

train_df = tf.data.Dataset.from_tensor_slices((ids,attributes))



# Display tensor values

for index, tensor in train_df.enumerate().as_numpy_iterator():

    if index > 5:

        break

    print('Image file path:', tensor[0], '  Labels:', tensor[1])
def decode_jpeg(filename, label):

    bits = tf.io.read_file(GCS_DS_PATH+'/train/'+filename+'.png')

    image = tf.image.decode_jpeg(bits)

    image = tf.image.resize_with_crop_or_pad(image,300,300)

    if tf.shape(image)[2] == 1:

        image = tf.image.grayscale_to_rgb(image)

    return image, label

image_ds = train_df.map(decode_jpeg)



# Display images and labels

for index, tensor in image_ds.enumerate().as_numpy_iterator():

    if index > 5:

        break

    print('Image Shape:',tensor[0].shape, '   Labels:', tensor[1])

#     if index == 3:

#         print('Image Data Example: ',tensor[0])
# TPU_CORES = strategy.num_replicas_in_sync

# IMAGE_SIZE = [512,512]

# EPOCHS = 12

BATCH_SIZE = 16

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

  ds = ds.prefetch(buffer_size=AUTO)



  return ds

train_ds = prepare_for_training(image_ds)

# TAKES VERY LONG TIME TO RUN, 10+ minutes

# image_batch, label_batch = next(iter(train_ds))
def get_Title(labels,index):

    title = ''

    tag_array = labels[index].split()

    for index, tag in enumerate(tag_array):

        tag_array[index] = CLASS_NAMES.iloc[int(tag)].attribute_name

    return ' '.join(tag_array)
def show_batch(image_batch, label_batch):

  plt.figure(figsize=(10,10))

  for n in range(16):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(image_batch[n])

      plt.title(get_Title(label_batch,n))

      plt.axis('off')

show_batch(image_batch.numpy(), label_batch.numpy())
MobileNetV2 = tf.keras.applications.MobileNetV2(input_shape=[300,300,3], include_top=False)

MobileNetV2.trainable = False



model = tf.keras.Sequential([

    MobileNetV2,

    tf.keras.layers.Conv2D(kernel_size=3, filters=24, padding="same", activation="relu"),

    tf.keras.layers.Conv2D(kernel_size=3, filters=24, padding="same", activation="relu"),

    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=12, padding="same", activation="relu"),

    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=6, padding="same", activation="relu"),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(5, activation="softmax")

])



model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)
model.fit(image_batch,label_batch,batch_size=16,epochs=1)