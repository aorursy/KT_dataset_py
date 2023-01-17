import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



VALID = 10000 # Validation data size



x = tf.placeholder(tf.float32, [None, 784])



W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))



train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()



data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe



labels = np.array(data.pop('label'))

labels = LabelEncoder().fit_transform(labels)[:, None]

labels = OneHotEncoder().fit_transform(labels).todense()



data = data / 256

train_data, valid_data = data[:-VALID], data[-VALID:]

train_labels, valid_labels = labels[:-VALID], labels[-VALID:]



for _ in range(1000):

  sess.run(train_step, feed_dict={x: train_data, y_: train_labels})

  

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: valid_data, y_: valid_labels}))
