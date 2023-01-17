# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/youtube8m-2019/frame-sample/frame"))
# Loading libraries & datasets

import tensorflow as tf

import numpy as np

from IPython.display import YouTubeVideo



frame_lvl_record = "../input/youtube8m-2019/frame-sample/frame/train00.tfrecord"

frame_lvl_record1 = "../input/youtube8m-2019/frame-sample/frame/train01.tfrecord"

valid_recordd = "../input/youtube8m-2019/validate-sample/validate/validate00.tfrecord"

for example in tf.python_io.tf_record_iterator(frame_lvl_record1):

  result = tf.train.Example.FromString(example)

  print(result.features.feature['segment_start_times'].int64_list.value)

  #print(result.features.feature['text_label'].bytes_list.value)
for example in tf.python_io.tf_record_iterator(frame_lvl_record):

  result = tf.train.Example.FromString(example)

  print(result.features.feature['labels'].int64_list.value)

  print(result.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
csv = pd.read_csv("/kaggle/input/youtube8m-2019/vocabulary.csv")

csv