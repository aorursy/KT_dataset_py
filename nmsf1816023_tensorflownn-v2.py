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
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np


Max_len = 15120

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if ((max_value - min_value) == 0):
            result[feature_name] = 0
        else:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)  ## 被除数为0
    return result


def label(train_y_N):
    array = [0, 0, 0, 0, 0, 0, 0]
    array[train_y_N[0] - 1] = 1
    for i in range(1, len(train_y_N)):
        new_array = [0, 0, 0, 0, 0, 0, 0]
        new_array[train_y_N[i] - 1] = 1
        array = np.row_stack((array, new_array))
    return array


def RecoverToNumber(train_y_L):
    array = []
    for tr_L in train_y_L:
        list = tr_L.tolist()
        max_index = list.index(max(list))+1
        array.append(max_index)
    return array


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    layer_name = 'layer1'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # add one more layer and return the output of this layer
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/weights', Weights)  # tensorflow >= 0.12
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
            tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs





train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')

# 将数据拆分成两列
train_x = train.iloc[:, 1:55]
train_y = label(train.Cover_Type)

# 对train_x test_x标准化
train_x_nmlz = normalize(train_x)
test_x = normalize(test.iloc[:,1::])

sess = tf.Session()

xs = tf.placeholder(tf.float32, [None, 54])
ys = tf.placeholder(tf.float32, [None, 7])  ## 为啥莫凡就可以写10？ 因为他的数据
prediction = add_layer(xs, 54, 7, activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess.run(init)
for i in range(1500):
    batch_xs, batch_ys = train_x_nmlz.iloc[(i * 100) % Max_len: (100 + i *100)%Max_len, :], train_y[(i * 100)%Max_len : (100 + i * 100)%Max_len]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})  ##
y_pre = sess.run(prediction, feed_dict={xs:test_x})
y_pre = RecoverToNumber(y_pre)
Id = test.Id
datafram = pd.DataFrame({'Id':Id,'Cover_Type':y_pre})
datafram.to_csv('res2.csv',index=False)


datafram