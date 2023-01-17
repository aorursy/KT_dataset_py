# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'/kaggle/input/characters-digits-train.csv', header=None)

test = pd.read_csv(r'/kaggle/input/characters-digits-test.csv' , header = None)

train.head()

test.head()

train['Output']  = np.where(train[0] < 10 ,0 ,1)

test['Output']  = np.where(test[0] < 10 ,0 ,1)
train['Output']
train_labels = train['Output']

train_data = train.drop(columns=['Output', 0])



test_labels = test['Output']

test_data = test.drop(columns=['Output', 0])

train_labels = pd.get_dummies(train_labels)

test_labels = pd.get_dummies(test_labels)

train_labels.head()

train_data
test_labels.head()
train_data = train_data.values

train_labels = train_labels.values

test_data = test_data.values

test_labels = test_labels.values

del train, test
import matplotlib.pyplot as plt

%matplotlib inline

def rotate(image):

    image = image.reshape([28, 28])

    image = np.fliplr(image)

    image = np.rot90(image)

    return image.reshape([28 * 28])

train_data = np.apply_along_axis(rotate, 1, train_data)/255

test_data = np.apply_along_axis(rotate, 1, test_data)/255

plt.imshow(train_data[4].reshape([28, 28]), cmap='Greys_r')

plt.show()
import tensorflow as tf
tf.reset_default_graph()

xs = tf.placeholder(tf.float32, [None, 784], name='input')

ys = tf.placeholder(tf.float32, [None, 2], name='exp_output')

dropout = tf.placeholder(tf.float32, name='dropout')
x_image = tf.reshape(xs, [-1, 28, 28, 1])

layer = tf.layers.conv2d(x_image, 64, [5,5], padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

layer = tf.layers.max_pooling2d(layer, pool_size=(2,2), strides=2) # [-1, 14, 14, 64]

layer = tf.layers.batch_normalization(layer)

layer = tf.layers.conv2d(layer, 128, [2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

layer = tf.layers.max_pooling2d(layer, pool_size=(2,2), strides=2) # [-1, 7, 7, 128]

x_flat = tf.reshape(layer, [-1, 7*7*128])

flatten = tf.layers.dense(x_flat, 1024, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

flatten = tf.nn.dropout(flatten, keep_prob=1-dropout)

flatten = tf.layers.dense(flatten, 512, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

flatten = tf.layers.batch_normalization(flatten)

flatten = tf.layers.dense(flatten, 128, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

flatten = tf.layers.dense(flatten, 2)

pred = tf.nn.softmax(flatten, name='output')

    
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(

    labels=ys,

    logits=flatten))











train = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct = tf.equal(tf.argmax(flatten, 1), tf.argmax(ys, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))











init = tf.global_variables_initializer()
NUM = 112800

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(20):

        for i in range(int(NUM / 100)):

            x_batches, y_batches = train_data[i * 100: (i + 1) * 100], train_labels[i * 100: (i + 1) * 100]

            sess.run(train, feed_dict={xs: x_batches, ys: y_batches, dropout: 0.5})

            

            if i % 1000 == 0:

                acc, entropy = sess.run([accuracy, cross_entropy], feed_dict={xs: test_data,

                                                    ys: test_labels,

                                                    dropout: 0})

                print('Train Entropy : ', sess.run(cross_entropy, feed_dict={xs: x_batches, ys: y_batches, dropout: 0.5}))

                print('Test Accr & Entropy : ', acc, entropy)

                # save_and_generate_proto(sess)

    acc = sess.run(accuracy, feed_dict={xs: test_data,

                                                ys: test_labels,

                                                dropout: 0})

    print(acc)