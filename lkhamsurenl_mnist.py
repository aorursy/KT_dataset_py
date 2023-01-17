# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # tensorflow cnn will be used to recognize digits



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv("../input/train.csv").sample(frac=1) # randomly shuffle

data_size = data.shape[0] # total number of examples 



train_labels = np.asarray([[1 if value == i else 0 for i in range(10)] for value in data["label"][:train_size]])

train_data = np.asarray([data.iloc[i].values[1:] for i in range(train_size)])

print("shape of train: {}".format(train_data.shape))

print("shape of train_labels: {}".format(train_labels.shape))



valid_labels = np.asarray([[1 if value == i else 0 for i in range(10)] for value in data["label"][train_size:]])

valid_data = np.asarray([data.iloc[i].values[1:] for i in range(train_size, data_size)])

print("shape of valid: {}".format(valid_data.shape))

print("shape of valid labels: {}".format(valid_labels.shape))



test = pd.read_csv("../input/test.csv")



test_data = np.asarray([test.iloc[i].values for i in range(test_size)])

print("shape of test: {}".format(test_data.shape))

# Any results you write to the current directory are saved as output.
# CONSTANTS

TRAIN_SIZE = data_size - int(data_size / 10)

TEST_SIZE = test.shape[0]

BATCH_SIZE = 50

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
# CNN layers

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')
# placeholders

x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder is used for swapping in training data for training

y_ = tf.placeholder(tf.float32, shape=[None, 10])



# 1st layer

W_conv1 = weight_variable([5,5,1,32])

b_conv1 = bias_variable([32])



x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)



# 2nd layer

W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)



# 3rd fully connected layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



# 4th output layer

W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])

# prediction

y = tf.matmul(h_fc1, W_fc2) + b_fc2



# start the session

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())



# loss function

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))



# evaluation setup

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train 

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(int(TRAIN_SIZE / BATCH_SIZE)):

    start = i * BATCH_SIZE

    end = (i + 1) * BATCH_SIZE

    train_batch = train_data[start: end]

    label_batch = train_labels[start: end]

    train_step.run(feed_dict={x: train_batch, y_: label_batch})

        

    # evaluation at each 50 th step to see if the progress is being made.

    if i % 20 == 0:

        print(accuracy.eval(feed_dict={x: train_batch, y_: label_batch}))

        

print(accuracy.eval(feed_dict={x: valid_data, y_: valid_labels}))
def evaluate():

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={x: valid_data, y_: valid_labels}))