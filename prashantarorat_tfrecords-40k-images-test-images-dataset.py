import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf, re, math

import random

import os
Cover_PATH = '../input/alaska2-image-steganalysis/Cover/'

JMiPOD_PATH = '../input/alaska2-image-steganalysis/JMiPOD/'

JUNIWARD_PATH = '../input/alaska2-image-steganalysis/JUNIWARD/'

UERD_PATH = '../input/alaska2-image-steganalysis/UERD/'

TEST_PATH = "../input/alaska2-image-steganalysis/Test/"



Cover = [Cover_PATH + i for i in os.listdir(Cover_PATH)][:10000]

JMiPOD = [JMiPOD_PATH +i for i in os.listdir(JMiPOD_PATH)][:10000]

UERD = [UERD_PATH+i for i in os.listdir(UERD_PATH)][:10000]

JUNIWARD = [JUNIWARD_PATH+i for i in os.listdir(JUNIWARD_PATH)][:10000]



print('There are %i images ' % len(Cover))

print('There are %i images ' % len(JMiPOD))

print('There are %i images ' % len(UERD))

print('There are %i images ' % len(JUNIWARD))
import pandas as pd

all_paths = Cover + JMiPOD + UERD + JUNIWARD



paths = pd.DataFrame(all_paths,columns=["image_path"])

paths['target'] = paths.image_path.apply(lambda x:x.split("/")[3])



dictt = {

    'Cover':0,

    'JMiPOD':1,

    'JUNIWARD':2,

    'UERD':3

}



paths['target'] = paths.target.apply(lambda x:dictt[x])
import seaborn as sns

sns.countplot(paths['target'])
from sklearn.utils import shuffle

paths = shuffle(paths).reset_index(drop=True)
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
paths
def serialize_example(feature0, feature1):

  feature = {

      'image': _bytes_feature(feature0),

      'target': _int64_feature(feature1),

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
SIZE = 10000

CT = paths.shape[0]//SIZE + int(paths.shape[0]%SIZE!=0)

for j in range(CT):

    print()

    print('Writing TFRecord %i of %i...'%(j,CT))

    CT2 = min(SIZE,paths.shape[0]-j*SIZE)

    with tf.io.TFRecordWriter('train%.2i-%i.tfrec'%(j,CT2)) as writer:

        for k in range(CT2):

            img = cv2.imread(paths.iloc[k]['image_path'])

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            example = serialize_example(

                img,

                paths.iloc[k]['target']

            )

            writer.write(example)

            if k%1000==0: print(k,', ',end='')
# test_paths = [TEST_PATH+i for i in os.listdir(TEST_PATH)]

# test_paths[1].split("/")[-1]
# def serialize_example_test(feature0,feature1):

#   feature = {

#       'image': _bytes_feature(feature0),

#       "file_name":_bytes_feature(feature1)

#   }

#   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

#   return example_proto.SerializeToString()





# SIZE = 5000

# CT = len(test_paths)//SIZE + int(len(test_paths)%SIZE!=0)

# for j in range(CT):

#     print()

#     print('Writing TFRecord %i of %i...'%(j,CT))

#     CT2 = min(SIZE,len(test_paths)-j*SIZE)

#     with tf.io.TFRecordWriter('test%.2i-%i.tfrec'%(j,CT2)) as writer:

#         for k in range(CT2):

#             img = cv2.imread(test_paths[k])

#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

#             img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

#             example = serialize_example_test(

#                 img,

#                 str.encode(test_paths[k].split("/")[-1])

#             )

#             writer.write(example)

#             if k%1000==0: print(k,', ',end='')