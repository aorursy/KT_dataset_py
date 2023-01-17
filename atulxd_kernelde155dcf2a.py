# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
n_observations = 10

feature0 = np.random.choice([1, 2], n_observations)



feature1 = np.random.randn(n_observations)

strings = np.array([b'cat',b'dog'])

feature2 = np.random.choice(strings, n_observations)

feature3 = np.random.randn(n_observations, 2, 2)

ds = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() 

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_example(feature0, feature1, feature2, feature3):

  feature = {

      'feature0': _int64_feature(feature0),

      'feature1': _float_feature(feature1),

      'feature2': _bytes_feature(feature2),

      'feature3': _bytes_feature(feature3),

  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()
for feature0, feature1, feature2, feature3 in ds.take(1):

  serialized_example = serialize_example(feature0, 

                                 feature1, 

                                 feature2, 

                                 tf.io.serialize_tensor(feature3))

  print(serialized_example)