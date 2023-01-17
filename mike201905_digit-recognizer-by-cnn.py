# import libraries

import tensorflow as tf

import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

# utilities

from datetime import datetime

# Disable warning output

import warnings

warnings.filterwarnings('ignore')
class ReadCsvData(object):



  def __init__(self):

    # Read data from train dataset

    df_train = pd.read_csv('../input/train.csv')



    # Read data from test dataset

    df_test = pd.read_csv('../input/test.csv')



    self._train = df_train.drop(['label'], axis=1).values

    self._labels = df_train['label'].values

    self._test = df_test.values



  def get_data(self):

    # Convert train and test data from [0, 255] -> [0.0, 1.0]

    self._train = self._train.astype(np.float32)

    self._train = np.multiply(self._train, 1.0 / 255.0)

    self._test = self._test.astype(np.float32)

    self._test = np.multiply(self._test, 1.0 / 255.0)

    

    # Convert label to one-hot presenting

    self._labels = np.identity(10)[self._labels]

    # Return data

    return self._train, self._labels,self._test
class DataSet(object):



  def __init__(self,

               images,

               labels):

    assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

    self._num_examples = images.shape[0]

    self._images = images

    self._labels = labels

    self._epochs_completed = 0

    self._index_in_epoch = 0



  def next_batch(self, batch_size):

    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch

    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:

      # Finished epoch

      self._epochs_completed += 1

      # Shuffle the data

      perm = np.arange(self._num_examples)

      np.random.shuffle(perm)

      self._images = self._images[perm]

      self._labels = self._labels[perm]

      # Start next epoch

      start = 0

      self._index_in_epoch = batch_size

      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]
# Initiate tensorflow session

sess = tf.InteractiveSession()



# placeholder

x = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



# variable

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))



# weights and bias variable: 1st layer

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 50], stddev=0.1))

b_conv1 = tf.Variable(tf.constant(0.1, shape=[50]))
# resharpe train data to 28*28 

# parameter:

# -1: number of image 28,28: size of each image 1: No. of channel

x_image = tf.reshape(x, [-1,28,28,1])



# convolutional and pool: 1st layer

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# weights and bias variable: 2nd layer

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 50, 100], stddev=0.1))

b_conv2 = tf.Variable(tf.constant(0.1, shape=[100]))



# convolutional and pool: 2nd layer

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# FULL connection and relu: 1st layer 

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 100, 1024], stddev=0.1))

b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*100])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout

rate = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, 1-rate)
# FULL connection : 2nd layer 

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))

b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# cross entropy

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))



# Train function : AdamOptimizer

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



# accuracy

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# prediction

predict = tf.argmax(y_conv,1)
# initiate variables of tensorflow

sess.run(tf.global_variables_initializer())
# Get input data and initiate DataSet

input_data=ReadCsvData()

x_train, y_label, x_test = input_data.get_data()
# Start train CNN

Batch_size=100

# Train_NUmber:18000 rate:0.5 Kaggle Accuracy:0.99242

# Train_NUmber:10000 rate:0.5 Kaggle Accuracy:0.99071

Train_Number=5300

accracies = []
print('Start Learning', datetime.now(),)

for j in range(3):

    train_dataset = DataSet(x_train,y_label)

    for i in range(Train_Number):

      batch_x, batch_y = train_dataset.next_batch(Batch_size)

      if i%100 == 0:

        train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, rate: 0.0})

        accracies.append(train_accuracy)

        print("step %d, training accuracy %g"%(i, train_accuracy))

      train_step.run(feed_dict={x: batch_x, y_: batch_y, rate: 0.5})



print("step %d, training accuracy %g"%(i, train_accuracy))

print('Finish Learning', datetime.now(),)
# Show the prediction

submission_file=pd.DataFrame({'ImageId':np.arange(1,(x_test.shape[0] + 1)), 'Label':predict.eval(feed_dict={x: x_test, rate: 0.0})})

print(submission_file)

submission_file.to_csv("submission_v1.csv", index=False)

print('Save submission', datetime.now(),)
saver = tf.train.Saver()

# saver.save(sess, 'mnist_model')

# saver.restore(sess, "mnist_model")

sess.close()