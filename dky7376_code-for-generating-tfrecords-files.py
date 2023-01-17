import numpy as np
import tensorflow as tf
import math
import pandas as pd
import glob
import os
def read_submission_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'test/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'sample_submission.csv')
    df['path'] = df['id'].map(mapping)
    df['label'] = -1
    df['prob'] = -1
    return df

def read_train_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)
    
    counts_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['prob'] = ((1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df, dict(zip(range(len(uniques)), uniques))


submission_df = read_submission_file('../input/landmark-recognition-2020/')
train_df, mapping = read_train_file('../input/landmark-recognition-2020/')

train_df
dataset = train_df.iloc[:1000,:]
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))
# code for generating tfrecord file
record_file = 'train00.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for i in range(len(dataset)):
    path = dataset.path[dataset.index[i]]
    label = dataset.label[dataset.index[i]]
    image_string = tf.io.read_file(path)
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())
def read_labeled_tfrecord(example):
    tfrec_format = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
    return tf.io.parse_single_example(example, tfrec_format)
ds = tf.data.TFRecordDataset('./train00.tfrecords')
ds = ds.map(read_labeled_tfrecord)
import IPython.display as display
for feature in ds.take(3):
  image_raw = feature['image_raw'].numpy()
  display.display(display.Image(data=image_raw))