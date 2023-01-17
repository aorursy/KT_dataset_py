!pip install tf-object-detection
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import argparse

from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def __split(df, group):
   data = namedtuple('data', ['filename', 'object'])
   gb = df.groupby(group)
   return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, class_dict):
   with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
      encoded_jpg = fid.read()
   encoded_jpg_io = io.BytesIO(encoded_jpg)
   image = Image.open(encoded_jpg_io)
   width, height = image.size

   filename = group.filename.encode('utf8')
   image_format = b'jpg'
   xmins = []
   xmaxs = []
   ymins = []
   ymaxs = []
   classes_text = []
   classes = []

   for index, row in group.object.iterrows():
      if set(['xmin_rel', 'xmax_rel', 'ymin_rel', 'ymax_rel']).issubset(set(row.index)):
         xmin = row['xmin_rel']
         xmax = row['xmax_rel']
         ymin = row['ymin_rel']
         ymax = row['ymax_rel']

      elif set(['xmin', 'xmax', 'ymin', 'ymax']).issubset(set(row.index)):
         xmin = row['xmin'] / width
         xmax = row['xmax'] / width
         ymin = row['ymin'] / height
         ymax = row['ymax'] / height

      xmins.append(xmin)
      xmaxs.append(xmax)
      ymins.append(ymin)
      ymaxs.append(ymax)
      classes_text.append(row['class'].encode('utf8'))
      classes.append(class_dict[row['class']])

   tf_example = tf.train.Example(features=tf.train.Features(
       feature={
           'image/height': dataset_util.int64_feature(height),
           'image/width': dataset_util.int64_feature(width),
           'image/filename': dataset_util.bytes_feature(filename),
           'image/source_id': dataset_util.bytes_feature(filename),
           'image/encoded': dataset_util.bytes_feature(encoded_jpg),
           'image/format': dataset_util.bytes_feature(image_format),
           'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
           'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
           'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
           'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
           'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
           'image/object/class/label': dataset_util.int64_list_feature(classes), }))
   return tf_example


def class_dict_from_pbtxt(pbtxt_path):
   # open file, strip \n, trim lines and keep only
   # lines beginning with id or display_name

   with open(pbtxt_path, 'r', encoding='utf-8-sig') as f:
      data = f.readlines()

   name_key = None
   if any('display_name:' in s for s in data):
      name_key = 'display_name:'
   elif any('name:' in s for s in data):
      name_key = 'name:'

   if name_key is None:
      raise ValueError(
          "label map does not have class names, provided by values with the 'display_name' or 'name' keys in the contents of the file"
      )

   data = [l.rstrip('\n').strip() for l in data if 'id:' in l or name_key in l]

   ids = [int(l.replace('id:', '')) for l in data if l.startswith('id')]
   names = [
       l.replace(name_key, '').replace('"', '').replace("'", '').strip() for l in data
       if l.startswith(name_key)]

   # join ids and display_names into a single dictionary
   class_dict = {}
   for i in range(len(ids)):
      class_dict[names[i]] = ids[i]

   return class_dict

def createTFRecords(class_dict, output_path, image_dir, csv_input):
    class_dict = class_dict_from_pbtxt(pbtxt_input)
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = __split(examples, 'filename')

    for group in tqdm(grouped, desc='groups'):
      tf_example = create_tf_example(group, path, class_dict)
      writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))





image_dir = '../input/test-dataset/imageset_833'
pbtxt_input = '../input/test-dataset/label_dict.txt'

# Divide em treinamento e teste
csv_input = '../input/test-dataset/imageset_833.csv'
data = pd.read_csv(csv_input)
train = data.sample(frac=0.7, random_state=1)
test = data.loc[~data.index.isin(train.index)]
train.to_csv("train.csv")
test.to_csv("test.csv")


csv_test = 'test.csv'
csv_train = 'train.csv'

output_test_path = 'test.records'
output_train_path = 'train.records'

createTFRecords(pbtxt_input, output_test_path, image_dir, csv_test)
createTFRecords(pbtxt_input, output_train_path, image_dir, csv_train)
import os
import pandas as pd

#print(os.path.exists('../input/test-dataset/imageset_833/wolves_test_a_000.png'))

csv_input = '../input/test-dataset/imageset_833.csv'
data = pd.read_csv(csv_input)

train = data.sample(frac=0.7, random_state=1)
test = data.loc[~data.index.isin(train.index)]

train.to_csv("train.csv")
test.to_csv("test.csv")


!git clone https://github.com/tensorflow/models.git
!cp models/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config faster_rcnn_inception_v2_coco.config 

!ls
!python models/research/object_detection/builders/model_builder_test.py
!protoc models/research/object_detection/protos/*.proto --python_out=.