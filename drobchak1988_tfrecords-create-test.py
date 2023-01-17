import cv2, math, os, sys

from tqdm import tqdm_notebook as tqdm

import zipfile

import io

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

warnings.filterwarnings("ignore")

PATH_IMAGE = '/kaggle/input/testimagesgraph/'
!mkdir /kaggle/working/test-tfrecords
GCS_OUTPUT ='/kaggle/working/test-tfrecords/' # prefix for output file names

SHARDS = 12

TARGET_SIZE = [64, 64]
def resize_and_crop_image(image, label):

  w = tf.shape(image)[0]

  h = tf.shape(image)[1]

  tw = TARGET_SIZE[1]

  th = TARGET_SIZE[0]

  resize_crit = (w * th) / (h * tw)

  image = tf.cond(resize_crit < 1,

                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true

                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false

                 )

  nw = tf.shape(image)[0]

  nh = tf.shape(image)[1]

  image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

  return image, label

  



def decode_jpeg_and_label(filename):

  bits = tf.io.read_file(filename)

  image = tf.image.decode_jpeg(bits)

  # parse flower name from containing directory

  label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')

  label = label.values[-1]

  return image, label



filenames = tf.data.Dataset.list_files(PATH_IMAGE + '*.png', seed=35155) # This also shuffles the images

dataset1 = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)

dataset1 = dataset1.map(resize_and_crop_image, num_parallel_calls=AUTO) 
def display_9_images_from_dataset(dataset):

  plt.figure(figsize=(13,13))

  subplot=331

  for i, (image, label) in enumerate(dataset):

    img = image.numpy()

    img = cv2.resize(img,(64,64))

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(img.astype('float32'))

    plt.title(label.numpy().decode("utf-8"), fontsize=16)

    subplot += 1

    if i==2:

        break

  plt.tight_layout()

  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.show()

display_9_images_from_dataset(dataset1)
def recompress_image(image, label):

  height = tf.shape(image)[0]

  width = tf.shape(image)[1]

  image = tf.cast(image, tf.uint8)

  image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

  return image, label, height, width

dataset3 = dataset1.map(recompress_image, num_parallel_calls=AUTO)

dataset3 = dataset3.batch(12) 
def _bytestring_feature(list_of_bytestrings):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))



def _int_feature(list_of_ints): # int64

  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def _float_feature(list_of_floats): # float32

  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

  



def to_tfrecord(tfrec_filewriter, img_bytes, label):  

  feature = {

      "image": _bytestring_feature([img_bytes]), 

      "label":         _bytestring_feature([label])

  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

  

print("Writing TFRecords")

for shard, (image, label, height, width) in enumerate(dataset3):

  # batch size used as shard size here

  shard_size = image.numpy().shape[0]

  # good practice to have the number of records in the filename

  filename = GCS_OUTPUT + "test-{:02d}-{}.tfrec".format(shard, shard_size)

  

  with tf.io.TFRecordWriter(filename) as out_file:

        for i in range(shard_size):

            example = to_tfrecord(out_file,

                            image.numpy()[i], # re-compressed image: already a byte string

                            label.numpy()[i])

            out_file.write(example.SerializeToString())

        print("Wrote file {} containing {} records".format(filename, shard_size))
import shutil

shutil.make_archive('test-frecords', 'zip', '/kaggle/working/test-tfrecords')
