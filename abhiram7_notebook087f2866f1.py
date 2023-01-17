import tensorflow as tf

import numpy as np

import csv

import random

import pandas as pd



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



# Write to the log:

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs



#my_data = np.genfromtxt('train.csv', delimiter=',')

#my_data_test = np.genfromtxt('test.csv', delimiter=',')

#row = len(my_data[:,1])

train = np.array(train)

test = np.array(test)

col = len(train[1,:])

print (col)

x_train = train[:,1:col]

x_test = test

y_train = train[:,0]

y_train = y_train.astype(int)

#print np.max(y_train)

#print x_train.shape, y_train.shape

y_train = np.eye(np.max(y_train) + 1)[y_train]

#print x_train.shape, y_train.shape



sess = tf.InteractiveSession()



x = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

    strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])



x_image = tf.reshape(x, [-1,28,28,1])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

batch_size = 10

train_size = 40000

x_train_test = x_train[train_size:41999,:]

y_train_test = y_train[train_size:41999,:]
for i in range(1):

#batch = mnist.train.next_batch(50)

    for j in range(0,(train_size-1),batch_size):

        x_batch = x_train[j:j+batch_size,:]

        y_batch = y_train[j:j+batch_size,:]

        if j%20000 == 0:

            train_accuracy = accuracy.eval(feed_dict={

                x:x_batch, y_: y_batch, keep_prob: 1.0})

            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.9})

    c= list(zip(x_batch[0:train_size,:],y_batch[0:train_size,:]))

    random.shuffle(c)

    x_batch, y_batch = zip(*c)

    x_batch = np.array(x_batch)

    y_batch = np.array(y_batch)

print ('training done')
print("test accuracy %g"%accuracy.eval(feed_dict={

    x: x_train_test, y_: y_train_test, keep_prob: 1.0}))

print("train accuracy %g"%accuracy.eval(feed_dict={

    x: x_train[0:train_size,:], y_: y_train[0:train_size,:], keep_prob: 1.0}))

#print("test accuracy %g"%accuracy.eval(feed_dict={

#    x: x_train_test[4000:8000,:], y_: y_train_test[4000:8000,:], keep_prob: 1.0}))



y_class = np.zeros(len(x_train[:,1]),dtype=np.int8)

print ('predicting new values')

for i in range(0,28000,4000):

    y_pred = tf.argmax(y_conv,1)

    y_class[i:i+4000] = sess.run(y_pred, feed_dict = \

        {x:x_test[i:i+4000, :],keep_prob:1.0})

    print (i)
print ('opening file for write')



label = ["ImageId,Label"]

with open('../input/sample_submission.csv', 'wb') as csvfile:

    spamwriter = csv.writer(csvfile, delimiter=' ')

    spamwriter.writerow(label)

    for i in xrange(1,28000+1):

        values = [str(i)+','+str(y_class[i-1])]

        spamwriter.writerow(values)

print ('file write complete')

#------------things to add to code--------------

#.validation check for differnt models

#.randomize inputs