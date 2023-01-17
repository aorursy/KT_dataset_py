from google.colab import auth
auth.authenticate_user()

import os

# TODO: Enter your own Google Cloud Project name here
os.environ["GCLOUD_PROJECT"] = "your-project-name"
!export GCLOUD_PROJECT=your-project-name
class FLAGS:
    test_directory = 'test/images/*/*/*/'
    # TODO: add path to your own GCS bucket folder here
    output_directory = 'gs://your-gcs-bucket/folder'
    test_csv_path = 'test.csv'
    num_shards_per_part = 2 # per TAR file
    seed = 0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf

_FILE_IDS_KEY = 'file_ids'
_IMAGE_PATHS_KEY = 'image_paths'
_LABELS_KEY = 'labels'
_TEST_SPLIT = 'test'
_TRAIN_SPLIT = 'train'
_VALIDATION_SPLIT = 'validation'


def _get_all_image_files_and_labels(name, csv_path, image_dir):
  """Process input and get the image file paths, image ids and the labels.

  Args:
    name: 'train' or 'test'.
    csv_path: path to the Google-landmark Dataset csv Data Sources files.
    image_dir: directory that stores downloaded images.
  Returns:
    image_paths: the paths to all images in the image_dir.
    file_ids: the unique ids of images.
    labels: the landmark id of all images. When name='test', the returned labels
      will be an empty list.
  Raises:
    ValueError: if input name is not supported.
  """
  image_paths = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
  file_ids = [os.path.basename(os.path.normpath(f))[:-4] for f in image_paths]
  if name == _TRAIN_SPLIT:
    with tf.io.gfile.GFile(csv_path, 'rb') as csv_file:
      df = pd.read_csv(csv_file)
    df = df.set_index('id')
    labels = [int(df.loc[fid]['landmark_id']) for fid in file_ids]
  elif name == _TEST_SPLIT:
    labels = []
  else:
    raise ValueError('Unsupported dataset split name: %s' % name)
  return image_paths, file_ids, labels


def _process_image(filename):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.jpg'.

  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  Raises:
    ValueError: if parsed image has wrong number of dimensions or channels.
  """
  # Read the image file.
  with tf.io.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = tf.io.decode_jpeg(image_data, channels=3)

  # Check that image converted to RGB
  if len(image.shape) != 3:
    raise ValueError('The parsed image number of dimensions is not 3 but %d' %
                     (image.shape))
  height = image.shape[0]
  width = image.shape[1]
  if image.shape[2] != 3:
    raise ValueError('The parsed image channels is not 3 but %d' %
                     (image.shape[2]))

  return image_data, height, width


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_id, image_buffer, height, width, label=None):
  """Build an Example proto for the given inputs.

  Args:
    file_id: string, unique id of an image file, e.g., '97c0a12e07ae8dd5'.
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    label: integer, the landmark id and prediction label.

  Returns:
    Example proto.
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  features = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/id': _bytes_feature(file_id.encode('utf-8')),
      'image/encoded': _bytes_feature(image_buffer)
  }
  if label is not None:
    features['image/class/label'] = _int64_feature(label)
  example = tf.train.Example(features=tf.train.Features(feature=features))

  return example


def _write_tfrecord(output_prefix, image_paths, file_ids, labels,
                    part_idx, total_parts, num_shards_per_part):
  """Read image files and write image and label data into TFRecord files.

  Args:
    output_prefix: string, the prefix of output files, e.g. 'train'.
    image_paths: list of strings, the paths to images to be converted.
    file_ids: list of strings, the image unique ids.
    labels: list of integers, the landmark ids of images. It is an empty list
      when output_prefix='test'.

  Raises:
    ValueError: if the length of input images, ids and labels don't match
  """
  if output_prefix == _TEST_SPLIT:
    labels = [None] * len(image_paths)
  if not len(image_paths) == len(file_ids) == len(labels):
    raise ValueError('length of image_paths, file_ids, labels shoud be the' +
                     ' same. But they are %d, %d, %d, respectively' %
                     (len(image_paths), len(file_ids), len(labels)))

  spacing = np.linspace(0, len(image_paths), num_shards_per_part + 1, dtype=np.int)

  for shard in range(num_shards_per_part):
    output_file = os.path.join(
        FLAGS.output_directory,
        '%s-%.5d-of-%.5d' % (output_prefix,
                             part_idx * num_shards_per_part + shard,
                             total_parts * num_shards_per_part))
    writer = tf.io.TFRecordWriter(output_file)
    print('    - Processing shard ', shard, ' and writing file ', output_file)
    for i in range(spacing[shard], spacing[shard + 1]):
      image_buffer, height, width = _process_image(image_paths[i])
      example = _convert_to_example(file_ids[i], image_buffer, height, width,
                                    labels[i])
      writer.write(example.SerializeToString())
    writer.close()


def _build_test_tfrecord_dataset(csv_path, image_dir, part_idx, total_parts):
  """Build a TFRecord dataset for the 'test' split.

  Args:
    csv_path: path to the 'test' Google-landmark Dataset csv Data Sources files.
    image_dir: directory that stores downloaded images.

  Returns:
    Nothing. After the function call, sharded TFRecord files are materialized.
  """
  image_paths, file_ids, labels = _get_all_image_files_and_labels(
      _TEST_SPLIT, csv_path, image_dir)
  _write_tfrecord(_TEST_SPLIT, image_paths, file_ids, labels,
                  part_idx, total_parts, FLAGS.num_shards_per_part)


def build_test_dataset(part_idx, total_parts):
  _build_test_tfrecord_dataset(FLAGS.test_csv_path, FLAGS.test_directory,
                               part_idx, total_parts)
import os
import glob
import hashlib
from datetime import datetime
import subprocess
import shutil
from urllib.request import urlretrieve


def download_to_gcs(split, indices):
    csv_url = f'https://s3.amazonaws.com/google-landmark/metadata/{split}.csv'
    print('  - Downloading csv {}...'.format(csv_url), end=' ')
    begin = datetime.now()
    urlretrieve(csv_url, f'{split}.csv')
    print('Done in', datetime.now() - begin)
    
    tar_format_url = 'https://s3.amazonaws.com/google-landmark/{split}/images_{idx:03d}.tar'
    md5_format_url = 'https://s3.amazonaws.com/google-landmark/md5sum/{split}/md5.images_{idx:03d}.txt'
    
    for idx in indices:
        print(f'{split}-{idx:03d}:')
        
        archive_url = tar_format_url.format(split=split, idx=idx)
        print('  - Downloading archive {}...'.format(archive_url), end=' ')
        begin = datetime.now()
        urlretrieve(archive_url, 'tmp_images.tar')
        print('Done in', datetime.now() - begin)

        checksum_url = md5_format_url.format(split=split, idx=idx)
        print('  - Downloading md5 checksum {}...'.format(checksum_url), end=' ')
        begin = datetime.now()
        urlretrieve(checksum_url, 'tmp_md5.txt')
        print('Done in', datetime.now() - begin)

        print('  - Calculating MD5 checksum...', end=' ')
        md5sum_call = subprocess.Popen(['md5sum', 'tmp_images.tar'], stdout=subprocess.PIPE)
        md5sum, _ = md5sum_call.communicate()
        md5sum_call.wait()
        md5sum = md5sum.decode('ascii').split()[0]

        with open('tmp_md5.txt', 'r') as f:
            md5sum_expected = f.read().split()[0]

        if not md5sum == md5sum_expected:
            print('Mismatch, the index will be added to `failed_indices.txt')
            raise ValueError
        else:
            print('Matched')

        print('  - Extracting archive to images folder...', end=' ')
        begin = datetime.now()
        if not os.path.isdir(f'{split}/images/'):
            os.makedirs(f'{split}/images/')
        extraction_task = subprocess.Popen(['tar', '-xf', './tmp_images.tar', '-C', f'{split}/images/'])
        extraction_task.wait()
        print('Done in', datetime.now() - begin)

        print('  - Building TFRecrod and extracting to GCS:')
        begin = datetime.now()
        if split == 'test':
            build_test_dataset(idx, len(indices))
        else:
            raise ValueError('split should be test')
        print('    Done in', datetime.now() - begin)
        
        print('  - Cleaning up working directory...', end=' ')
        begin = datetime.now()
        os.remove('tmp_images.tar')
        os.remove('tmp_md5.txt')
        shutil.rmtree(f'{split}/')
        print('Done in', datetime.now() - begin)
    
    print('Finished successfully')
download_to_gcs('test', [idx for idx in range(20)])