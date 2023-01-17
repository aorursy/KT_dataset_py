import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# converting data to np arrays
y = tf.Session().run(tf.one_hot(np.array(train['label']), depth=10))
X = np.array(train.drop('label', axis=1))
# splitting data into test and train
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=1)
# splitting train data into batches
X_train = np.split(X_train, 294)
y_train = np.split(y_train, 294)

batch_no=0
def next_batch():
    global batch_no
    ret_val = (X_train[batch_no], y_train[batch_no])
    batch_no = (batch_no + 1) % 294
    return ret_val
# building the tensorflow graph
Xin = tf.placeholder(dtype=tf.float32, shape=[None,784])
W01 = tf.Variable(tf.random_normal(shape=[784, 32]))
B01 = tf.Variable(tf.random_normal(shape=[32]))
V_1 = tf.nn.sigmoid(tf.add(B01, tf.matmul(Xin, W01)))
W12 = tf.Variable(tf.random_normal(shape=[32,28]))
B12 = tf.Variable(tf.random_normal(shape=[28]))
V_2 = tf.nn.sigmoid(tf.add(B12, tf.matmul(V_1, W12)))
W23 = tf.Variable(tf.random_normal(shape=[28,10]))
B23 = tf.Variable(tf.random_normal(shape=[10]))
Yout = tf.nn.sigmoid(tf.add(B23, tf.matmul(V_2, W23)))
Ytrue = tf.placeholder(dtype=tf.float32, shape=[None,10])
# parameters
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Yout, labels=Ytrue))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)
init = tf.global_variables_initializer()
# running the network
with tf.Session() as ss:
    ss.run(init)
    for i in range(1000):
        Xt, yt = next_batch()
        ss.run(optimizer, feed_dict = {Xin : Xt, Ytrue : yt})
        if (i % 100 == 0):
            pred = tf.equal(tf.argmax(Yout, 1), tf.argmax(y_test, 1))
            accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
            print(i//100, ss.run(accuracy, feed_dict={Xin : X_test, Ytrue : y_test}))
