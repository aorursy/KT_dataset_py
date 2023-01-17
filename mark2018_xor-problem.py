# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import tensorflow as tf

data = tf.placeholder(tf.float32, shape=(4, 2))
label = tf.placeholder(tf.float32, shape=(4, 1))

with tf.variable_scope('layer1') as scope:
  weight = tf.get_variable(name='weight', shape=(2, 2))
  bias = tf.get_variable(name='bias', shape=(2,))
  x = tf.nn.sigmoid(tf.matmul(data, weight) + bias)
with tf.variable_scope('layer2') as scope:
  weight = tf.get_variable(name='weight', shape=(2, 1))
  bias = tf.get_variable(name='bias', shape=(1,))
  x = tf.matmul(x, weight) + bias

# 正则化
preds = tf.nn.sigmoid(x)
# 损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=x))
# 学习率
learning_rate = tf.placeholder(tf.float32)
# 优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 训练数据,定义学习率
train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_label = np.array([[0], [1], [1], [0]])
lr = 0.05

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(50000):
    # 训练若干轮
        _, l, pred = sess.run([optimizer, loss, preds], feed_dict={ data: train_data, label: train_label, learning_rate: lr })
        if step % 500 == 0:
            print('Step: {} -> Loss: {} -> Predictions: \n{}'.format(step, l, pred))
