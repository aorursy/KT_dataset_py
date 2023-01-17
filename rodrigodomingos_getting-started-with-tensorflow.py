import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from subprocess import check_output



print(check_output(["ls", "../input"]).decode("utf8"))



from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../input/", one_hot=True)



print(mnist)



x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))



train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()



for _ in range(1000):

 # batch_xs, batch_ys = mnist.train.next_batch(100)

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


