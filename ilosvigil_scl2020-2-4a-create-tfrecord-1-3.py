import tensorflow as tf 

import numpy as np

import os

import glob

import pandas as pd

import PIL

import gc

from PIL import Image
print(f'Numpy version : {np.__version__}')

print(f'Pandas version : {pd.__version__}')

print(f'Tensorflow version : {tf.__version__}')

print(f'Pillow version : {PIL.__version__}')
!ls /kaggle/input
df_train = pd.read_parquet('/kaggle/input/csv-with-cleaned-ocr-text/train.parquet', engine='pyarrow').sort_values("filename").reset_index(drop=True)
df_test = pd.read_parquet('/kaggle/input/csv-with-cleaned-ocr-text/test.parquet', engine='pyarrow')

df_test
def _bytes_feature(value):

  """Returns a bytes_list from a string / byte."""

  if isinstance(value, type(tf.constant(0))):

    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _list_float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _list_int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
RESIZE_WIDTH = 512

RESIZE_HEIGHT = 512



TFRECORD_MAX_SIZE = 80 * 1024 * 1024 # 80 MB



TOTAL_IMAGES = len(df_train.index)

# TOTAL_IMAGES = len(df_test.index)



# part 1 : 0:TOTAL_IMAGES // 2 (train) [CURRENT]

# part 2 : TOTAL_IMAGES // 2:TOTAL_IMAGES (train)

# part 3 : 0:TOTAL_IMAGES (test)

START_INDEX = 0

END_INDEX = TOTAL_IMAGES // 2



BATCH_IMAGE = 1024
def create_tfrecord(index, df):

    index = str(index).zfill(3)

    curr_file = f"train-{index}.tfrecords"

    writer = tf.io.TFRecordWriter(curr_file)

    for index, row in df.iterrows():

        category_str = str(row['category']).zfill(2)



        image = f'/kaggle/input/shopee-product-detection-student/train/train/train/{category_str}/{row["filename"]}'

        img = open(image, 'rb')

        img_read = img.read()

        image_decoded = tf.image.decode_jpeg(img_read, channels=3)

        resized_img = tf.image.resize_with_pad(image_decoded,target_width=RESIZE_WIDTH,target_height=RESIZE_HEIGHT,method=tf.image.ResizeMethod.BILINEAR)

        resized_img = tf.cast(resized_img,tf.uint8)

        resized_img = tf.io.encode_jpeg(resized_img)



        feature = {

            'filename': _bytes_feature(tf.compat.as_bytes(row['filename'])),

            'label': _int64_feature(row['category']),

            'words': _list_float_feature(row['words']),

            'image': _bytes_feature(resized_img),

            'height' : _int64_feature(RESIZE_HEIGHT),

            'width' : _int64_feature(RESIZE_WIDTH)

        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
for i in range(START_INDEX, END_INDEX, BATCH_IMAGE):

    print(f'Create TFRecords #{i // BATCH_IMAGE}')

    if i + BATCH_IMAGE < END_INDEX:

        create_tfrecord(i // BATCH_IMAGE, df_train.loc[i:i+BATCH_IMAGE])

    else:

        create_tfrecord(i // BATCH_IMAGE, df_train.loc[i:END_INDEX])

    gc.collect()
!ls -lah