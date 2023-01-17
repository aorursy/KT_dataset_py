# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np

import sklearn

import sklearn.preprocessing

import tensorflow as tf

# Any results you write to the current directory are saved as output.
def one_hot(i, m=None):

    m = i if m is None else m

    arr = np.zeros([m+1])

    arr[m-i] = 1

    return arr



def row_to_sample(row):

    s = one_hot(int(row[0]), 1)

    age = np.array([row[1]/70.0])

    party = one_hot(int(row[2]), 1)

    time = one_hot(int(row[3])-1, 2)

    city = one_hot(int(row[4]), 3)

    work = one_hot(int(row[5]), 2)

    occupation = one_hot(int(row[6]), 3)

    married= one_hot(int(row[7])-1, 1)

    children = np.array([row[8]/4.0])

    popularity = np.array([row[9]/2.0])

    

    y = np.array([row[10]/100.0])

    

    x = np.concatenate( [s,age,party,time,city,work,occupation,married,children,popularity] )

    

    x = np.expand_dims(x, 0)

    y = np.expand_dims(y, 0)

    return x,y

train_df = pd.read_csv('../input/train.csv', sep=' ')

test_df = pd.read_csv('../input/test.csv', sep=' ')
train_data = [ row_to_sample(x) for idx, x in train_df.iterrows() ]

test_data = [ row_to_sample(x) for idx, x in test_df.iterrows() ]

X = tf.placeholder(dtype=tf.float32, shape=[None, 23])

Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

dp = tf.placeholder(dtype=tf.float32)



net = tf.layers.dense(inputs=X, units=23, activation=tf.nn.relu)

net = tf.layers.dropout(inputs=net, rate=dp)

net = tf.layers.dense(inputs=net, units=21, activation=tf.nn.relu)

net = tf.layers.dropout(inputs=net, rate=dp)

net = tf.layers.dense(inputs=net, units=4, activation=tf.nn.relu)

net = tf.layers.dense(inputs=net, units=1)



loss = tf.losses.mean_squared_error(net, Y)



op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
epochs = 2001

ststas_every_n = 100

sess = tf.Session()

sess.run(tf.global_variables_initializer())

train_loss = []

test_loss = []



for i in range(epochs):

    for x,y  in train_data:

        _, _loss = sess.run([op, loss], feed_dict={X:x,Y:y,dp:0.6})

        train_loss.append(_loss)



    for x,y  in test_data:

        _loss = sess.run(loss, feed_dict={X:x,Y:y, dp:0})

        test_loss.append(_loss)

    if i%ststas_every_n==0:

        print('{} | Train loss: {:0.5f} Test loss: {:0.5f}'.format(i, np.mean(train_loss), np.mean(test_loss)))



pred = []

gt = []

for x,y  in test_data:

    _pred = sess.run(net, feed_dict={X:x,Y:y, dp:0})

    print('Predicted {} | Real {}'.format(_pred, y))

    pred.append(_pred[0][0])

    gt.append(y[0][0])
from matplotlib import pyplot

import numpy



bins = numpy.linspace(0,3,4)



pyplot.scatter(bins, pred, label='Prediction')

pyplot.scatter(bins, gt, label='GroundTrugth')

pyplot.legend(loc='upper right')

pyplot.show()