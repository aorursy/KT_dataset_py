import os

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from glob import glob

import io

from PIL import Image
photo_filepaths = glob(f"/kaggle/input/gan-getting-started/photo_tfrec/*.tfrec")



monet_filepaths = glob(f"/kaggle/input/gan-getting-started/monet_tfrec/*.tfrec")
# Create a dictionary describing the features.

image_feature_description = {

    'image': tf.io.FixedLenFeature([], tf.string),

}
def _parse_image_function(example_proto):

  # Parse the input tf.train.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, image_feature_description)

raw_image_dataset = tf.data.TFRecordDataset(monet_filepaths)

monet_image_dataset = raw_image_dataset.map(_parse_image_function)



raw_photo_image_dataset = tf.data.TFRecordDataset(photo_filepaths)

photo_image_dataset = raw_photo_image_dataset.map(_parse_image_function)
print('Display Monet image...')

for e in monet_image_dataset.take(1):

    image = Image.open(io.BytesIO(e['image'].numpy()))

image
print('Display Photo image...')

for e in photo_image_dataset.take(1):

    image = Image.open(io.BytesIO(e['image'].numpy()))

image