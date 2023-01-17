# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# read the csv data
Mnist_data = pd.read_csv("../input/fashion-mnist_train.csv")
Mnist_test = pd.read_csv("../input/fashion-mnist_test.csv")
print(Mnist_data.head())

# Get the Training data
sess = tf.InteractiveSession()
Train_x = Mnist_data.loc[:, 'pixel1':'pixel784']
Train_y = Mnist_data.loc[:, 'label']

Test_x = Mnist_test.loc[:, 'pixel1':'pixel784']
Test_y = Mnist_test.loc[:, 'label']
# Set y one-hot vector
Train_y_one_hot = sess.run(tf.one_hot(Train_y, depth=10, name="One_hot_op"))
Test_y_one_hot = sess.run(tf.one_hot(Test_y, depth=10, name='testonehot_op'))
print(Train_x.head())
print(Test_y_one_hot[0:3])
# print (Train_x.head())
# print (Train_y.head())
# with tf.Session() as sess:
#     print (sess.run(Train_y_one_hot))
print(Train_x.iloc[1:4,:])
# Let's struct the model first.
num_epoch = 600
with tf.name_scope('Variable'):
    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
#     drop_out = tf.placeholder(tf.float32)

# build the model
with tf.name_scope('Layer1'):
    W1 = tf.Variable(initial_value=tf.truncated_normal([784,300], stddev=0.1), name='W1')
    b1 = tf.Variable(tf.zeros([300]), name='b1')
    A1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    
with tf.name_scope('Layer2'):
    W2 = tf.Variable(initial_value=tf.truncated_normal([300,100], stddev=0.1), name='W2')
    b2 = tf.Variable(tf.zeros([100]), name='b2')
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
    
with tf.name_scope('Output'):
    W3 = tf.Variable(initial_value=tf.truncated_normal([100, 10], stddev=0.1), name='W3')
    b3 = tf.Variable(tf.zeros([10]), name='b3')
    Z3 = tf.matmul(A2, W3) + b3
    y_prediction = tf.nn.softmax(Z3)
    

# loss function
with tf.name_scope('loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits = Z3, name='loss')
# Accuracy function
with tf.name_scope('Accuracy'):
    Correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(y,1))
    Accuracy = tf.reduce_mean(tf.cast(Correct_prediction,dtype=tf.float32))
# train step
with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# run the model
init = tf.global_variables_initializer()   
sess.run(init)
for i in range(50):
    for epoch in range(num_epoch):
        batch_x = Train_x.iloc[epoch*100:(epoch+1)*100-1, :]
        batch_y = Train_y_one_hot[epoch*100:(epoch+1)*100-1, :]
        sess.run(train_step, feed_dict={X:batch_x, y:batch_y})
       
    print(sess.run(Accuracy, feed_dict={X:Test_x, y:Test_y_one_hot}))
#             print(sess.run(Correct_prediction, feed_dict={X:Test_x, y:Test_y_one_hot}))
