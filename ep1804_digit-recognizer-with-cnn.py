import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random



import tensorflow as tf
d1 = pd.read_csv("../input/train.csv")
print(d1.shape)

d1.head()
# ref: https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/run/164725

def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return pd.DataFrame(labels_one_hot)



d2 = d1.sample(frac=1)



x_tr = d2.iloc[:38000, 1:]

x_ob = d2.iloc[38000:, 1:]



y_tr = dense_to_one_hot(d2.iloc[:38000, 0], 10)

y_ob = dense_to_one_hot(d2.iloc[38000:, 0], 10)
x_tr.head()
y_tr.head()
print(x_tr.shape)

print(y_tr.shape)

print(x_ob.shape)

print(y_ob.shape)
# parameters

learning_rate = 0.001

data_size = x_tr.shape[0]

batch_size = 200



print(data_size)
# input place holders

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)
# weight initialization

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



# convolution

def conv(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# reshape input into tensor

x1 = tf.reshape(x, [-1, 28, 28, 1])



# convolution layer

W1 = weight_variable([5, 5, 1, 32])

b1 = bias_variable([32])

a_con_1 = tf.nn.relu(conv(x1, W1) + b1) # (?, 28, 28, 32)

a_max_1 = max_pool(a_con_1) # (?, 14, 14, 32)

a_out_1 = tf.nn.dropout(a_max_1, keep_prob = keep_prob)



# convolution layer

W2 = weight_variable([5, 5, 32, 64])

b2 = bias_variable([64])

a_con_2 = tf.nn.relu(conv(a_out_1, W2) + b2) # (?, 14, 14, 64)

a_max_2 = max_pool(a_con_2) # (?, 7, 7, 64)

a_out_2 = tf.nn.dropout(a_max_2, keep_prob = keep_prob)



# fully-connected layer

a_flat_2 = tf.reshape(a_out_2, [-1, 7 * 7 * 64]) # (?, 3136)

W3 = weight_variable([7 * 7 * 64, 1024])

b3 = bias_variable([1024])

a3 = tf.nn.relu(tf.matmul(a_flat_2, W3) + b3) # (?, 1024)

a_out_3 = tf.nn.dropout(a3, keep_prob = keep_prob)



# softmax layer

W4 = weight_variable([1024, 10])

b4 = bias_variable([10])

z4 = tf.matmul(a_out_3, W4) + b4

a4 = tf.nn.softmax(z4) # (?, 10)
print(a_con_1.get_shape())

print(a_max_1.get_shape())

print(a_con_2.get_shape())

print(a_max_2.get_shape())

print(a3.get_shape())

print(a4.get_shape())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z4, labels=y))

#cost = -tf.reduce_sum(y * tf.log(a4))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Prediction and check accuracy

correct = tf.equal(tf.argmax(a4, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# variables for next_batch function

next_batch__x = x_tr

next_batch__y = y_tr

next_batch__data_size = data_size

next_batch__batch_size = batch_size

next_batch__start = 0

next_batch__epoch_done = False

next_batch__epochs_completed = 0



# serve data by batches

def next_batch():

    

    global next_batch__x

    global next_batch__y

    global next_batch__data_size

    global next_batch__batch_size

    global next_batch__start

    global next_batch__epoch_done

    global next_batch__epochs_completed    

    

    next_batch__epoch_done = False

    

    # end of epoch

    if next_batch__start > next_batch__data_size:

        # shuffle the data

        arr = np.arange(next_batch__data_size)

        np.random.shuffle(arr)

        next_batch__x = next_batch__x.iloc[arr, :]

        next_batch__y = next_batch__y.iloc[arr, :]

        # end epoch

        next_batch__start = 0

        next_batch__epoch_done = True

        next_batch__epochs_completed += 1



    start = next_batch__start

    end = start + next_batch__batch_size

    next_batch__start = end

    return next_batch__x[start:end], next_batch__y[start:end]
# Initialize tensorflow session

sess = tf.Session()

sess.run(tf.global_variables_initializer())



# Initialize next_batch function

next_batch__start = 0

next_batch__epochs_completed = 0



# Train

while(next_batch__epochs_completed < 5):

    x_batch, y_batch = next_batch()

    sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})

    if (next_batch__epoch_done):

        c_b, ac_b = sess.run([cost, accuracy], feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

        c, ac = sess.run([cost, accuracy], feed_dict={x: x_ob, y: y_ob, keep_prob: 1.0})

        print('Epochs: %2d, Batch cost: %f, accuracy: %f, Test cost: %s, accuracy: %f' % 

              (next_batch__epochs_completed, c_b, ac_b, c, ac))

print('Learning Finished!')