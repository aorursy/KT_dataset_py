# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # tensorflow learning library (1.4)

from sklearn.model_selection import train_test_split



input_file = pd.read_csv('../input/kc_house_data.csv')

data_frame = pd.DataFrame(input_file)
data = data_frame.drop(['price', 'date', 'id'], axis=1)

data[0:5]
target = data_frame['price']

target[0:5]
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.4, random_state=42)
sess = tf.InteractiveSession()

feature_count = len(X_train.columns)



# Define data

x = tf.placeholder(tf.float64, shape=[None, feature_count])

y_ = tf.placeholder(tf.float64, shape=[len(Y_train),])
W = tf.Variable(tf.zeros([feature_count, 1]))

b = tf.Variable(tf.zeros([1,]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
X_train.shape
for _ in range(1000):

    train_step.run(feed_dict={x: X_train, y_: Y_train})