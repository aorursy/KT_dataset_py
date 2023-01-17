# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sess = tf.InteractiveSession()
i = tf.constant([
                 [1.0, 1.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
k = tf.constant([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0]
        ], dtype=tf.float32)

kernel = tf.reshape(k,[3,3,1,1], name = 'Kernel')
image = tf.reshape(i, [1,4,5,1], name = 'image')
res = tf.squeeze(tf.nn.conv2d(image, kernel, strides = [1,1,1,1], padding = "VALID"))
sess.run(res)
res = tf.squeeze(tf.nn.conv2d(image, kernel, strides = [1,1,1,1], padding = "SAME"))
sess.run(res)
inp = tf.constant([
    [
     [[1.0], [0.2], [2.0]],
     [[0.1], [1.2], [1.4]],
     [[1.1], [0.4], [0.4]]
    ] 
  ])

kernel = [1, 2, 2, 1]
max_pool = tf.nn.max_pool(inp, kernel, [1,1,1,1], "VALID")
sess.run(max_pool)
avg_pool = tf.nn.avg_pool(inp, kernel, [1, 1, 1, 1], "VALID")
sess.run(avg_pool)