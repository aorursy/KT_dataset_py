# LOAD LIBRARIES

import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf, re, math
# PATHS TO IMAGES

PATH = '../input/monet-ext-data/monet_dataset'

IMGS = os.listdir(PATH)

print('There are %i monet images '%(len(IMGS)))
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
def serialize_example(feature0, feature1, feature2):

  feature = {

      'image_name': _bytes_feature(feature0),

      'image': _bytes_feature(feature1),

      'target': _bytes_feature(feature2)

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
SIZE = 300

CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)

for j in range(CT):

    print(); print('Writing TFRecord %i of %i...'%(j,CT))

    CT2 = min(SIZE,len(IMGS)-j*SIZE)

    with tf.io.TFRecordWriter('monet_%.2i-%i.tfrec'%(j,CT2)) as writer:

        for k in range(CT2):

            img = cv2.imread("../input/monet-ext-data/monet_dataset/"+IMGS[SIZE*j+k])

            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            name = IMGS[SIZE*j+k].split('.')[0]

            example = serialize_example(

                str.encode(name),

                img, 

                b'monet')

            writer.write(example)

            if k%100==0: print(k,', ',end='')