"""A deep MNIST classifier using convolutional layers.

based on tutorials https://www.tensorflow.org/get_started/mnist/pros

and https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist

see Legal Notes at the end of the notebook

"""

# Disable linter warnings to maintain consistency with tutorial.

# pylint: disable=invalid-name

# pylint: disable=g-bad-import-order



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import random



from tensorflow.contrib.learn.python.learn.datasets import base

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet



import tensorflow as tf



SOURCE_URL = 'https://storage.googleapis.com/cloud-deeplearning/kaggle_mnist_data/'

DOWNLOAD_DATASETS=False

DATA_DIR = '../input/'

KAGGLE_TRAIN_CSV = 'train.csv'

KAGGLE_TEST_CSV = 'test.csv'

SUBMISSION_FILE = 'submission_mnist_dnn.csv'



# should sum up to 42000, the total number of images in train.csv

TRAIN_SIZE = 38000

VALID_SIZE = 2000

TEST_SIZE = 2000
def deepnn(x):

  """deepnn builds the graph for a deep net for classifying digits.

  Args:

    x: an input tensor with the dimensions (N_examples, 784), where 784 is the

    number of pixels in a standard MNIST image.

  Returns:

    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values

    equal to the logits of classifying the digit into one of 10 classes (the

    digits 0-9). keep_prob is a scalar placeholder for the probability of

    dropout.

  """

  # Reshape to use within a convolutional neural net.

  # Last dimension is for "features" - there is only one here, since images are

  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

  x_image = tf.reshape(x, [-1, 28, 28, 1])



  # First convolutional layer - maps one grayscale image to 32 feature maps.

  W_conv1 = weight_variable([5, 5, 1, 32])

  b_conv1 = bias_variable([32])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)



  # Pooling layer - downsamples by 2X.

  h_pool1 = max_pool_2x2(h_conv1)



  # Second convolutional layer -- maps 32 feature maps to 64.

  W_conv2 = weight_variable([5, 5, 32, 64])

  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



  # Second pooling layer.

  h_pool2 = max_pool_2x2(h_conv2)



  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image

  # is down to 7x7x64 feature maps -- maps this to 1024 features.

  W_fc1 = weight_variable([7 * 7 * 64, 1024])

  b_fc1 = bias_variable([1024])



  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



  # Dropout - controls the complexity of the model, prevents co-adaptation of

  # features.

  keep_prob = tf.placeholder(tf.float32)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



  # Map the 1024 features to 10 classes, one for each digit

  W_fc2 = weight_variable([1024, 10])

  b_fc2 = bias_variable([10])



  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob





def conv2d(x, W):

  """conv2d returns a 2d convolution layer with full stride."""

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')





def max_pool_2x2(x):

  """max_pool_2x2 downsamples a feature map by 2X."""

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')





def weight_variable(shape):

  """weight_variable generates a weight variable of a given shape."""

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)





def bias_variable(shape):

  """bias_variable generates a bias variable of a given shape."""

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)

# Can't use datasets from tensorflow tutorial, because the slicing of train, valid and test data differs from kaggle datasets

# TEMP_DIR = '/tmp/tensorflow/mnist/input_data'

# mnist = input_data.read_data_sets(TEMP_DIR, one_hot=True)
def custom_kaggle_mnist():

    """

    downloads and parses mnist train dataset for kaggle digit recognizer

    parsing and one_hot copied https://www.kaggle.com/kakauandme/tensorflow-deep-nn

    """

    if DOWNLOAD_DATASETS:

        base.maybe_download(KAGGLE_TRAIN_CSV, DATA_DIR, SOURCE_URL + KAGGLE_TRAIN_CSV)



    # Import data from datasource, see https://www.kaggle.com/kakauandme/tensorflow-deep-nn

    # read training data from CSV file 

    data = pd.read_csv(DATA_DIR + KAGGLE_TRAIN_CSV)

    

    from sklearn.utils import shuffle

    data = shuffle(data, random_state=42)

    

    images = data.iloc[:,1:].values

    images = images.astype(np.float)



    # convert from [0:255] => [0.0:1.0]

    images = np.multiply(images, 1.0 / 255.0)



    print('number of images in downloaded train dataset: {0[0]}'.format(images.shape))

    

    labels_flat = data.iloc[:,0].values

    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):

        num_labels = labels_dense.shape[0]

        index_offset = np.arange(num_labels) * num_classes

        labels_one_hot = np.zeros((num_labels, num_classes))

        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot



    labels = dense_to_one_hot(labels_flat, labels_count)

    labels = labels.astype(np.uint8)

    

    # split data into training & validation

    mnist_train_images = images[:TRAIN_SIZE]

    mnist_train_labels = labels[:TRAIN_SIZE]

    print('number of train images: {0[0]}'.format(mnist_train_images.shape))



    mnist_valid_images = images[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]

    mnist_valid_labels = labels[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]

    print('number of valid images: {0[0]}'.format(mnist_valid_images.shape))



    mnist_test_images = images[TRAIN_SIZE + VALID_SIZE:images.shape[0]]

    mnist_test_labels = labels[TRAIN_SIZE + VALID_SIZE:images.shape[0]]

    print('number of test images: {0[0]}'.format(mnist_test_images.shape))

    

    options = dict(dtype=np.float, reshape=False, seed=42)



    train = DataSet(mnist_train_images, mnist_train_labels, options)

    valid = DataSet(mnist_valid_images, mnist_valid_labels, options)

    test = DataSet(mnist_test_images, mnist_test_labels, options)



    return base.Datasets(train=train, validation=valid, test=test)
# Import data

mnist = custom_kaggle_mnist()
# Create the model

x = tf.placeholder(tf.float32, [None, 784])



# Define loss and optimizer

y_ = tf.placeholder(tf.float32, [None, 10])



# Build the graph for the deep net

y_conv, keep_prob = deepnn(x)



cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



saver = tf.train.Saver()
if DOWNLOAD_DATASETS:

    kaggle_test_file = base.maybe_download(KAGGLE_TEST_CSV, DATA_DIR, SOURCE_URL + KAGGLE_TEST_CSV)



# kaggle test data

# test_kaggle = (pd.read_csv(kaggle_test_file).values).astype('float32')

test_kaggle = (pd.read_csv(DATA_DIR + KAGGLE_TEST_CSV).values).astype('float32')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # 20000 with gpu

    for i in range(1000):

        batch = mnist.train.next_batch(50)

        if i % 100 == 0:

            train_accuracy = accuracy.eval(feed_dict={

                x: batch[0], y_: batch[1], keep_prob: 1.0})

            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



    print('validation accuracy %g' % accuracy.eval(feed_dict={

            x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))



    prediction_kaggle = tf.argmax(y_conv, 1)

    predictions = []

    pred_batch = 100

    for i in range(int(len(test_kaggle)/pred_batch)):

        feed_dict = {x : test_kaggle[i*pred_batch:(i+1)*pred_batch], keep_prob: 1.0}

        predictions.extend(sess.run(prediction_kaggle, feed_dict))

        if i % 50 == 0:

            print('{} images predicted.'.format(i*pred_batch))

    print('{} images predicted.'.format(len(test_kaggle)))
with open(SUBMISSION_FILE, 'w') as submission:

  submission.write('ImageId,Label\n')

  for index, prediction in enumerate(predictions):

    submission.write('{0},{1}\n'.format(index + 1, prediction))

  print("prediction submission written to {0}".format(SUBMISSION_FILE))
# train with checkpoint



# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     # 20000

#     for i in range(100):

#         batch = mnist.train.next_batch(50)

#         if i % 100 == 0:

#             train_accuracy = accuracy.eval(feed_dict={

#                 x: batch[0], y_: batch[1], keep_prob: 1.0})

#             print('step %d, training accuracy %g' % (i, train_accuracy))

#         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



#     print('validation accuracy %g' % accuracy.eval(feed_dict={

#             x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))



#     saver.save(sess, './model/dnn.mnist.model.ckpt')

# predict from checkpoint



# with tf.Session() as session:

#     session.run(tf.global_variables_initializer())

#     saver.restore(session, './model/dnn.mnist.model.ckpt')

#     prediction_kaggle = tf.argmax(y_conv, 1)

#     predictions = []

#     pred_batch = 100

#     for i in range(int(len(test_kaggle)/pred_batch)):

#         feed_dict = {x : test_kaggle[i*pred_batch:(i+1)*pred_batch], keep_prob: 1.0}

#         predictions.extend(session.run(prediction_kaggle, feed_dict))

#         if i % 50 == 0:

#             print('{} images predicted.'.format(i*pred_batch))

#     print('{} images predicted.'.format(len(test_kaggle)))
# Legal Notes:



# Copyright and License

# from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py:



# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================