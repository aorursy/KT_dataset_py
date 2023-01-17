# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data

import tensorflow as tf

#print(check_output(["ls", "input"]).decode("utf8"))

from time import time 



%matplotlib inline
tfrecords_filename = 'input/tfrecord/img_{}.tfrecords'

opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename.format(0), options=opts)

img_string = ""

for string_record in record_iterator:

    example = tf.train.Example()

    example.ParseFromString(string_record)

    height = int(example.features.feature['height']

                                 .int64_list

                                 .value[0])

    

    width = int(example.features.feature['width']

                                .int64_list

                                .value[0])

    

    img_string = (example.features.feature['img_raw'].bytes_list.value[0])

    

    category_id = (example.features.feature['category_id']

                                .int64_list

                                .value[0])



    product_id = (example.features.feature['product_id']

                                .int64_list

                                .value[0])

    

    print(height, width, category_id, product_id)

    if img_string != "":

        break

    
plt.imshow(imread(io.BytesIO(img_string)))