# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# inputing the data

from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

images = train.iloc[:,1:]

labels = train.iloc[:,0]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

print(test_images.shape)

print(train_images.shape)

print(test_labels.shape)

print(train_labels.shape)
# one hot encoding of labels

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)

num_classes = train_labels.shape

num_classes
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init = tf.global_variables_initializer()

#sess = tf.Session()

#sess.run(init)

sess = tf.InteractiveSession()

# Train

tf.global_variables_initializer().run()

print(train_images.shape)

print(train_labels.shape)



def getBatch(i, size, trainFeatures, trainLabels):

    startIndex = (i * size) % trainFeatures.shape[0]

    endIndex = startIndex + size

    batch_X = trainFeatures[startIndex : endIndex]

    batch_Y = trainLabels[startIndex : endIndex]

    return batch_X, batch_Y



for i in range(2000):

  batch_xs, batch_ys = getBatch(i,4200,train_images,train_labels)

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(test_images.shape)

print(train_images.shape)

print(test_labels.shape)

print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))