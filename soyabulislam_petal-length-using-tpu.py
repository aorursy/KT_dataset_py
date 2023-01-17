# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math, re, os

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import IPython.display as display

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

AUTO= tf.data.experimental.AUTOTUNE

import glob

                     

train_file= glob.glob('../input/tpu-getting-started/tfrecords-jpeg-512x512/train/*.tfrec')
val_file=glob.glob('../input/tpu-getting-started/tfrecords-jpeg-512x512/val/*.tfrec')

test_file= glob.glob('../input/tpu-getting-started/tfrecords-jpeg-512x512/test/*.tfrec')
ignore_order=tf.data.Options()

ignore_order.experimental_deterministic=False

train_data= tf.data.TFRecordDataset(train_file, num_parallel_reads=AUTO)

train_data=train_data.with_options(ignore_order)

training_features={

    'class':tf.io.FixedLenFeature([], tf.int64),

    'id':tf.io.FixedLenFeature([], tf.string),

    'image':tf.io.FixedLenFeature([], tf.string),

}
def train_feature_desc(example):

    return tf.io.parse_single_example(example, training_features)
train_id=[]

train_class=[]

train_image=[]



for i in train_file:

    train_image_data= tf.data.TFRecordDataset(i)

    

    train_image_data= train_image_data.map(train_feature_desc)

    ids=[str(id_features['id'].numpy())[2:-1] for id_features in train_image_data]

    train_id= train_id+ids





    classes= [str(class_features['class'].numpy()) for class_features in train_image_data]

    train_class=train_class+ classes





    images= [str(image_features['image'].numpy()) for image_features in train_image_data]

    train_image=train_image+ images

    

    

    
display.display(display.Image(data=train_image[211]))