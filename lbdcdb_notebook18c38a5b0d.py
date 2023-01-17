# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.info()

df_test.info()
batch_size =100



# Data as vectors

batches = df_train.ix[:,1:].values

validation_data = batches[:int(len(batches)*0.1)]

batches = batches[int(len(batches)*0.1):]

test_data = df_test.ix[:,1:].values



# One hot encoding of labels

labels = pd.get_dummies(df_train.ix[:,0]).astype(int).values

validation_labels = labels[:int(len(labels)*0.1)]

labels = labels[int(len(labels)*0.1):]

test_labels = pd.get_dummies(df_test.ix[:,0]).astype(int).values



# Batching

batches = iter(batches.reshape(-1,batch_size,784))

labels = iter(labels.reshape(-1,batch_size,10))
# Inputs

x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])



# Variables

W1 = tf.Variable(tf.zeros([784, 32]))

b1 = tf.Variable(tf.zeros([32]))

W2 = tf.Variable(tf.zeros([32, 10]))

b2 = tf.Variable(tf.zeros([10]))
# Graph

y1 = tf.nn.softmax(tf.matmul(x, W1) + b1)

y1 = tf.nn.relu(y1)

y = tf.nn.softmax(tf.matmul(y1, W2) + b2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
while True:

    try:

        batch_xs, batch_ys = next(batches), next(labels)

    except StopIteration:

        break

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Evaluation

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels}))