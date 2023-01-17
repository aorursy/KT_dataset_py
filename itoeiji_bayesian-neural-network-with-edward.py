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
data = pd.read_csv('../input/mushrooms.csv')

data.head()
data.shape
data.isnull().sum()
data.dtypes
data2 = pd.get_dummies(data)

data2.shape
data2['class_e'].sum() / data.shape[0] # class rate
data_x = data2.loc[:, 'cap-shape_b':].as_matrix().astype(np.float32)

data_y = data2.loc[:, :'class_p'].as_matrix().astype(np.float32)



N = 7000

train_x, test_x = data_x[:N], data_x[N:]

train_y, test_y = data_y[:N], data_y[N:]



in_size = train_x.shape[1]

out_size = train_y.shape[1]



EPOCH_NUM = 5

BATCH_SIZE = 1000



# for bayesian neural network

train_y2 = np.argmax(train_y, axis=1)

test_y2 = np.argmax(test_y, axis=1)
import sys

from tqdm import tqdm

import tensorflow as tf



x_ = tf.placeholder(tf.float32, shape=[None, in_size])

y_ = tf.placeholder(tf.float32, shape=[None, out_size])



w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), dtype=tf.float32)

b = tf.Variable(tf.constant(0.1, shape=[out_size]), dtype=tf.float32)

y_pre = tf.matmul(x_, w) + b



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pre))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess = tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in tqdm(range(EPOCH_NUM), file=sys.stdout):

    perm = np.random.permutation(N)

    for i in range(0, N, BATCH_SIZE):

        batch_x = train_x[perm[i:i+BATCH_SIZE]]

        batch_y = train_y[perm[i:i+BATCH_SIZE]]

        train_step.run(session=sess, feed_dict={x_: batch_x, y_: batch_y})

    acc = accuracy.eval(session=sess, feed_dict={x_: train_x, y_: train_y})

    test_acc = accuracy.eval(session=sess, feed_dict={x_: test_x, y_: test_y})

    if (epoch+1) % 1 == 0:

        tqdm.write('epoch:\t{}\taccuracy:\t{}\tvaridation accuracy:\t{}'.format(epoch+1, acc, test_acc))
import edward as ed

from edward.models import Normal, Categorical



x_ = tf.placeholder(tf.float32, shape=(None, in_size))

y_ = tf.placeholder(tf.int32, shape=(BATCH_SIZE))



w = Normal(loc=tf.zeros([in_size, out_size]), scale=tf.ones([in_size, out_size]))

b = Normal(loc=tf.zeros([out_size]), scale=tf.ones([out_size]))

y_pre = Categorical(tf.matmul(x_, w) + b)



qw = Normal(loc=tf.Variable(tf.random_normal([in_size, out_size])), scale=tf.Variable(tf.random_normal([in_size, out_size])))

qb = Normal(loc=tf.Variable(tf.random_normal([out_size])), scale=tf.Variable(tf.random_normal([out_size])))



y = Categorical(tf.matmul(x_, qw) + qb)



inference = ed.KLqp({w: qw, b: qb}, data={y_pre: y_})

inference.initialize()



sess = tf.Session()

sess.run(tf.global_variables_initializer())



with sess:

    samples_num = 100

    for epoch in tqdm(range(EPOCH_NUM), file=sys.stdout):

        perm = np.random.permutation(N)

        for i in range(0, N, BATCH_SIZE):

            batch_x = train_x[perm[i:i+BATCH_SIZE]]

            batch_y = train_y2[perm[i:i+BATCH_SIZE]]

            inference.update(feed_dict={x_: batch_x, y_: batch_y})

        y_samples = y.sample(samples_num).eval(feed_dict={x_: train_x})

        acc = (np.round(y_samples.sum(axis=0) / samples_num) == train_y2).mean()

        y_samples = y.sample(samples_num).eval(feed_dict={x_: test_x})

        test_acc = (np.round(y_samples.sum(axis=0) / samples_num) == test_y2).mean()

        if (epoch+1) % 1 == 0:

            tqdm.write('epoch:\t{}\taccuracy:\t{}\tvaridation accuracy:\t{}'.format(epoch+1, acc, test_acc))