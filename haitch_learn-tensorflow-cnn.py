import datetime

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # version 1.4.0

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
# we have X.npy and Y.npy as our input

# 5547 50*50 RGB images

inputX = np.load('../input/X.npy')

# array of 5547 numbers

originY = np.load('../input/Y.npy')

# transform originY to oneHot encoding.

inputY=np.zeros((5547,2))

inputY[np.arange(5547), originY] = 1
# Switch to code and run to see sample images

for i in range(2750, 2780):

    plt.imshow(inputX[i])

    plt.axis('off')

    plt.title('IDC' if originY[i] else "not IDC")

    plt.figure(figsize=(8, 2))
# we keep X_test and y_test not to be seen during training.

X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.15, random_state=42)

x=tf.placeholder(tf.float32, shape=[None,inputX[0].shape[0],inputX[0].shape[1],3])
def deepnn(x):

  """ build computation graph for image classification.

  Args:

    x: an input tensor with the dimensions (n, 50, 50, 3), represent n 50*50 RGB images.

  Returns:

    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values

    equal to the logits of classifying the digit into one of 10 classes (the

    digits 0-9). keep_prob is a scalar placeholder for the probability of

    dropout.

  """

  # First convolutional layer - maps image to 32 feature maps.

  with tf.name_scope('conv1'):

    W_conv1 = weight_variable([5, 5, 3, 32]) # sample (5,5), channel 3

    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

  print ("conv1 {}", h_conv1.shape)

  # Pooling layer - downsamples by 2X.

  with tf.name_scope('pool1'):

    h_pool1 = max_pool_3x3(h_conv1)

  print ("pool1 {}", h_pool1.shape)

  # Second convolutional layer -- maps 32 feature maps to 64.

  with tf.name_scope('conv2'):

    W_conv2 = weight_variable([5, 5, 32, 32])

    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  print ("conv2 {}", h_conv2.shape)

  # Second pooling layer.

  with tf.name_scope('pool2'):

    h_pool2 = max_pool_3x3(h_conv2)

  print ("pool2 {}", h_pool2.shape)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image

  # is down to 7x7x64 feature maps -- maps this to 1024 features.

  with tf.name_scope('fc1'):

    W_fc1 = weight_variable([6 * 6 * 32, 64])

    b_fc1 = bias_variable([64])



    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 32])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  print ("fc1 {}", h_fc1.shape)

  # Dropout - controls the complexity of the model, prevents co-adaptation of

  # features.

  with tf.name_scope('dropout'):

    keep_prob = tf.placeholder(tf.float32)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



  # Map the 1024 features to 10 classes, one for each digit

  with tf.name_scope('fc2'):

    W_fc2 = weight_variable([64, 2])

    b_fc2 = bias_variable([2])



    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  print ("final fc2 {}", y_conv.shape)

  return y_conv, keep_prob





def conv2d(x, W):

  """conv2d returns a 2d convolution layer with full stride."""

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')





def max_pool_2x2(x):

  """max_pool_2x2 downsamples a feature map by 2X."""

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')



def max_pool_3x3(x):

  """max_pool_3x3 downsamples a feature map by 2X."""

  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],

                        strides=[1, 3, 3, 1], padding='SAME')



def weight_variable(shape):

  """weight_variable generates a weight variable of a given shape."""

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)





def bias_variable(shape):

  """bias_variable generates a bias variable of a given shape."""

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)
y_ = tf.placeholder(tf.float32, [None, 2])

y_conv, keep_prob = deepnn(x)
with tf.name_scope('loss'):

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,

                                                            logits=y_conv)

cross_entropy = tf.reduce_mean(cross_entropy)



with tf.name_scope('adam_optimizer'):

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



with tf.name_scope('accuracy'):

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

  correct_prediction = tf.cast(correct_prediction, tf.float32)

  accuracy = tf.reduce_mean(correct_prediction)
import tempfile

graph_location = tempfile.mkdtemp()

print('Saving graph to: %s' % graph_location)

train_writer = tf.summary.FileWriter(graph_location)

train_writer.add_graph(tf.get_default_graph())
batch = 128

epoch = 150

epoch_index = 0



with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  kf = KFold(n_splits = epoch)

  for trainSet, validationSet in kf.split(X_train):

    batch_count = math.ceil(trainSet.shape[0]/batch)

    print('epoch %d start at %s with batchCount %d' % (epoch_index, datetime.datetime.now(), batch_count))

    index = 0

    for i in range(batch_count):

      train_step.run(feed_dict={x: X_train[trainSet][index : index + batch], y_: y_train[trainSet][index : index + batch], keep_prob: 0.50})

      index = index + batch

    validation_accuracy = accuracy.eval(feed_dict={

        x: X_train[validationSet], y_: y_train[validationSet], keep_prob: 1.0})

    print('epoch %d end at %s, training accuracy %g' % (epoch_index, datetime.datetime.now(), validation_accuracy))

    epoch_index = epoch_index + 1

  print('test accuracy %g' % accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})) 

# keep_prob: 0.50, batch: 64, epoch: 50, test accuracy 0.719088, takes 22 minutes

# keep_prob: 0.50, batch: 128, epoch: 50, test accuracy 0.690276, takes 22 minutes

# keep_prob: 0.40, batch: 64, epoch: 50, test accuracy 0.717887, takes 22 minutes

# keep_prob: 0.50, batch: 64, epoch: 100, test accuracy 0.747899, takes 50 minutes

# keep_prob: 0.50, batch: 64, epoch: 200, test accuracy 0.751501, takes 90 minutes

# keep_prob: 0.60, batch: 64, epoch: 150, test accuracy 0.735894, takes 50 minutes

# keep_prob: 0.55, batch: 64, epoch: 150, test accuracy 0.717887, takes 63 minutes

# keep_prob: 0.50, batch: 128, epoch: 150, test accuracy 0.732293, takes 58 minutes