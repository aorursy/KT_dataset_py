# Import libraries

import numpy as np

import tensorflow as tf

import pandas as pd
# Data preparation

iris = pd.read_csv('../input/Iris.csv')

iris = iris.drop('Id', 1)

iris = pd.get_dummies(iris, prefix='', prefix_sep='', columns=['Species'])

iris = iris.iloc[np.random.permutation(len(iris))]



index = int(iris.shape[0] * 0.8)

train_features = iris[:index]

train_target = pd.DataFrame({'Iris-setosa': train_features.pop('Iris-setosa'),

    'Iris-versicolor': train_features.pop('Iris-versicolor'),

    'Iris-virginica': train_features.pop('Iris-virginica')})

test_features = iris[index:]

test_target = pd.DataFrame({'Iris-setosa': test_features.pop('Iris-setosa'),

    'Iris-versicolor': test_features.pop('Iris-versicolor'),

    'Iris-virginica': test_features.pop('Iris-virginica')})
# Tensorflow variables

x = tf.placeholder(tf.float32, [None, 4])

W = tf.Variable(tf.zeros([4, 3]))

b = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 3])
# Tensorflow operations

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
# Training

sess = tf.Session()

sess.run(init)



for i in range(10000):

    if i%1000 == 0:

        print(sess.run(accuracy, feed_dict={x: test_features, y_: test_target}))

    sess.run(train_step, feed_dict={x: train_features, y_: train_target})
# Testing

print(sess.run(accuracy, feed_dict={x: test_features, y_: test_target}))