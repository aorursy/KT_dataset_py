# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

import time



batch_size = 32

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with tf.device('/gpu:0'):

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 256, 256, 3])

    y = tf.layers.conv2d(x, 16, (3, 3), (2, 2), 'same')

    y = tf.layers.conv2d(y, 32, (3, 3), (2, 2), 'same')

    y = tf.nn.relu(y)

    y = tf.layers.conv2d(y, 64, (3, 3), (2, 2), 'same')

    y = tf.layers.conv2d(y, 64, (3, 3), (2, 2), 'same')

    y = tf.nn.relu(y)

    y = tf.layers.conv2d(y, 4, (3, 3), (2, 2), 'same')

    y = tf.reshape(y, [-1, 64*4])

    y = tf.nn.softmax(y)

print(y)

sess.run(tf.global_variables_initializer())

t_beg = time.time()

cnt = 0

while cnt < 100:

    cnt += 1

    _y = sess.run(y, feed_dict={x:np.random.uniform(0,1.0, [batch_size, 256, 256, 3])})

t_end = time.time()

print(_y)

print(t_end - t_beg)

sess.close()

tf.reset_default_graph()
# An idea to convert all supervised learning into unsupervised learning! Learning is applied by inter-circuts.
