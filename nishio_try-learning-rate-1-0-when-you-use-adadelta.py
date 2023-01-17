# -*- encoding: utf-8 -*-

import tensorflow as tf

import numpy as np

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.preprocessing import OneHotEncoder



rng = np.random.RandomState(1234)

random_state = 42



## Placeholders

x = tf.placeholder(tf.float32, [None, 784])

t = tf.placeholder(tf.float32, [None, 10])



## Variables 

L1_SIZE = 200

W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(784, L1_SIZE)).astype('float32'), name='W1')

b1 = tf.Variable(np.zeros(L1_SIZE).astype('float32'), name='b1')



L2_SIZE = 50

W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(L1_SIZE, L2_SIZE)).astype('float32'), name='W2')

b2 = tf.Variable(np.zeros(L2_SIZE).astype('float32'), name='b2')



W3 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(L2_SIZE, 10)).astype('float32'), name='W3')

b3 = tf.Variable(np.zeros(10).astype('float32'), name='b3')



eta = tf.Variable(0.2, name='eta')



params = [W1, b1, W2, b2, W3, b3]



# graph

u1 = tf.matmul(x, W1) + b1

z1 = tf.nn.sigmoid(u1)

u2 = tf.matmul(z1, W2) + b2

z2 = tf.nn.sigmoid(u2)

u3 = tf.matmul(z2, W3) + b3

z3 = tf.nn.softmax(u3)

y = z3



# cost

cost = -tf.reduce_mean(

    tf.reduce_sum(

        t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), 

        axis=[1]

    )

)



valid = tf.argmax(y, 1)



# update rules

UPDATE_RULES = dict(

    Adadelta1=tf.train.AdadeltaOptimizer(learning_rate=0.2),

    Adadelta2=tf.train.AdadeltaOptimizer(learning_rate=1.0),

    Adam=tf.train.AdamOptimizer(learning_rate=0.2, epsilon=1.0),

)



# MNIST

import gzip, pickle, sys

f = gzip.open('../input/mnist.pkl.gz', 'rb')

(train_X, train_y), (valid_X, valid_y), _ = pickle.load(f, encoding="bytes")



ohe = OneHotEncoder(10, sparse=False)

train_y = ohe.fit_transform(train_y[:, np.newaxis])

valid_y = ohe.fit_transform(valid_y[:, np.newaxis])



n_epochs = 10

batch_size = 100

n_batches = train_X.shape[0] // batch_size



for rule in sorted(UPDATE_RULES):

    print(rule)

    train = UPDATE_RULES[rule].minimize(cost, var_list=params)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):

            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)

            for i in range(n_batches):

                start = i * batch_size

                end = start + batch_size

                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})

            pred_y, = sess.run([valid], feed_dict={x: valid_X, t: valid_y})

            score = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')

            print("EPOCH:: {}, F1: {:.3f}".format(

                epoch + 1, score

            ))