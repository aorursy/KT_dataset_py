# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



path = '../input/test3.csv'

# Any results you write to the current directory are saved as output.

df_orig = pd.read_csv(path, nrows=10000)

df_orig.head()

df = pd.get_dummies(df_orig)

df

import tensorflow as tf

import matplotlib.gridspec as gridspec

from pylab import rcParams



# Import data

# Any results you write to the current directory are saved as output.

filename_queue = tf.train.string_input_producer([path])



reader = tf.TextLineReader(skip_header_lines=1)

key, value = reader.read(filename_queue)



# Default values, in case of empty columns. Also specifies the type of the

# decoded result.

record_defaults = [[0],[0.0],[0],[0],[0],[0],[0.0],[0.0],[0.0],[0.0],[0]]

cols = tf.decode_csv(value, record_defaults=record_defaults)

target = tf.one_hot(cols[10], 4)

f1 = tf.expand_dims(tf.to_float(cols[0]), axis=0)

f2 = tf.expand_dims(tf.to_float(cols[1]), axis=0)

f3 = tf.one_hot(cols[2],3)

f4 = tf.one_hot(cols[3],4)

f5 = tf.one_hot(cols[4],5)

f6 = tf.one_hot(cols[5],5)

print(f1.shape)

print(f2.shape)

print(f3.shape)

print(f4.shape)

print(f5.shape)

print(f6.shape)

features = tf.concat([f1,f2,f3,f4,f5,f6], 0)

print(target)

print(target.shape)

print(features)

print(features.shape)
batch_size = 100

min_after_dequeue = 10000

capacity = min_after_dequeue + 3 * batch_size



example_batch, label_batch = tf.train.shuffle_batch(

    [features, target], batch_size=batch_size, capacity=capacity,

    min_after_dequeue=min_after_dequeue)



test_example_batch, test_label_batch = tf.train.shuffle_batch(

    [features, target], batch_size=batch_size*10, capacity=min_after_dequeue + 30 * batch_size,

    min_after_dequeue=min_after_dequeue)



# Create the model

x = tf.placeholder(tf.float32, [None, 19])

W = tf.Variable(tf.truncated_normal([19, 4],stddev=0.1))

b = tf.Variable(tf.truncated_normal([4],stddev=0.1))

y = tf.matmul(x, W) + b



y_ = tf.placeholder(tf.float32, [None, 4])



cross_entropy = tf.reduce_mean(

  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:

    tf.global_variables_initializer().run()



    # Start populating the filename queue.

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)



    # Train

    for i in range(10001):

        batch_xs, batch_ys = sess.run([example_batch, label_batch])

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i%100==0:

            test_batch_xs, test_batch_ys = sess.run([test_example_batch, test_label_batch])

            print(batch_ys)

            print(test_batch_ys)

            #test_batch_xs, test_batch_ys = sess.run([example_batch, label_batch])

            print("batch ",i," : ",sess.run(accuracy, feed_dict={x: test_batch_xs,

                                      y_: test_batch_ys}))





    coord.request_stop()

    coord.join(threads)