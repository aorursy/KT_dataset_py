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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
import tensorflow as tf



print("TODO: update input layer size")

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 99])



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(1000):

    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print(check_output(["ls", "../input/images"]).decode("utf8"))
filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png'])



reader = tf.WholeFileReader()

key, value = reader.read(filename_queue)



my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

import logging

logging.getLogger().setLevel(logging.INFO)



import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler





#read data set

train_file = "../input/train.csv"

test_file = "../input/test.csv"

train = pd.read_csv(train_file)

test = pd.read_csv(test_file)





x_train = train.drop(['species', 'id'], axis=1).values

le = LabelEncoder().fit(train['species'])

y_train = le.transform(train['species'])



x_test = test.drop(['id'], axis=1).values



scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)





# Build 3 layer DNN with 1024, 512, 256 units respectively.

classifier = tf.contrib.learn.DNNClassifier(hidden_units=[1024,512,256],

n_classes=99)



# Fit model.

classifier.fit(x=x_train, y=y_train, steps = 2000)



# Make prediction for test data

y = classifier.predict(x_test)

y_prob = classifier.predict_proba(x_test)



# prepare csv for submission

test_ids = test.pop('id')

submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)

submission.to_csv('submission_log_reg.csv')

submission.head()