# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

class Dataset(object):

    def __init__(self, data):

        self.rows = len(data.values)

        self.images = data.iloc[:,1:].values

        self.images = self.images.astype(np.float32)

        self.images = np.multiply(self.images, 1.0 / 255.0)

        self.labels = np.array([np.array([int(i == l) for i in range(10)]) for l in data.iloc[:,0].values]) #one-hot

        self.index_in_epoch = 0

        self.epoch = 0

    def next_batch(self, batch_size):

        start = self.index_in_epoch

        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.rows:

            self.epoch += 1

            perm = np.arange(self.rows)

            np.random.shuffle(perm)

            self.images = self.images[perm]

            self.labels = self.labels[perm]

            #next epoch

            start = 0

            self.index_in_epoch = batch_size

        end = self.index_in_epoch

        return self.images[start:end] , self.labels[start:end]
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_data = Dataset(train.iloc[0:37000])

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)



for i in range(1000):

    batch_xs, batch_ys = train_data.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



validate_data = Dataset(train.iloc[37000:])

print(sess.run(accuracy, feed_dict={x: validate_data.images, y_: validate_data.labels}))

import matplotlib.pyplot as plt

plt.imshow(validate_data.images[0].reshape(28,28)) 

print(np.argmax(sess.run(y, feed_dict={x:validate_data.images[:1]})))
test_images = test.values.astype(np.float32)

test_images = np.multiply(test_images, 1.0 / 255.0)



predictions = sess.run(y, feed_dict={x:test_images})

predictions = [np.argmax(p) for p in predictions]



result = pd.DataFrame({'ImageId': range(1,len(predictions)+1), 'Label': predictions})

result.to_csv('result.csv', index=False, encoding='utf-8')