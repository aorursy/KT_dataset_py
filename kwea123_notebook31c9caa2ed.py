%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf



datasource = pd.read_csv('../input/train.csv', delimiter=',')

X = np.array(datasource)

y = X[:,0]

X = X[:,1:]
fig = plt.figure()

for i in range(16):

    ax = fig.add_subplot(4, 4, i + 1)

    ax.set_xticks(())

    ax.set_yticks(())

    ax.imshow(X[i].reshape(28, 28), cmap='Greys_r')
def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)



def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)
def conv2d(x, W):

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, shape=[None, 784])

x_image = tf.reshape(x, [-1,28,28,1])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

enc.fit([[i] for i in range(10)])

enc.transform(np.transpose([y])).toarray()