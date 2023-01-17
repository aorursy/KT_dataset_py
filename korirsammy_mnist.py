#Import impotant libraries

import argparse

import sys

import tensorflow as tf



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
#Check input data

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
IMAGE_WIDTH = IMAGE_HEIGHT = 28

IMAGE_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT

LABELS = 10
#Iport data

data_train = pd.read_csv('../input/train.csv')
labels = np.array(data_train.pop('label')) # Remove the labels as a numpy array from the dataframe

labels = np.array([np.arange(LABELS) == label for label in labels])

data = np.array(data_train, dtype=np.float32)/255.0
plt.plot(np.argmax(labels[0:200], axis=1))
def showImage(image_data):

    plt.imshow(image_data.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))

    plt.axis('off')

    plt.show()

    

for i in range(1,4):

    showImage(data[[i]])
# Model input and output

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])

# Model parameters

W = tf.Variable(tf.zeros([IMAGE_SIZE, LABELS]))

b = tf.Variable(tf.zeros([LABELS]))



# Model Implementation

y = tf.nn.softmax(tf.matmul(x, W) + b)



# Define loss and optimizer

y_ = tf.placeholder(tf.float32, [None, 10])



# implement the cross-entropy function

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

evolution = []

for i in range(10000):

    row_selection = np.random.permutation(40000)[0:100]

    train_step.run(feed_dict={x: data[row_selection], y_: labels[row_selection]})

    evolution += [cross_entropy.eval(feed_dict={x: data[row_selection], y_: labels[row_selection]})]

    

plt.plot(evolution)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: data[40000:], y_: labels[40000:]}))
predictions = tf.argmax(y,1).eval(session=sess, feed_dict={x:data[100:200]})

for i in range(3):

    showImage(data[200+i])

    print( "Prediction of img #{0} is {1}".format(200+i, predictions[i]))
#Weights Initialization

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
 #Convolution and Pooling

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                          strides=[1, 2, 2, 1], padding='SAME')
#First Convolutional Layer

W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,IMAGE_WIDTH,IMAGE_HEIGHT,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)
#Second Convolutional Layer

W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
#Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#Readout Layer
W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())



for i in range(20000):

    row_selection = np.random.permutation(20000)[0:100]

    batch = data[row_selection]

    if i%100 == 0:

            train_accuracy = accuracy.eval(feed_dict={

                x:batch[0], y_: batch[1], keep_prob: 1.0})

            print("step %d, training accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={

     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))













    