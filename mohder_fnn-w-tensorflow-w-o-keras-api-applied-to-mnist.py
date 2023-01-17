# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data

X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

print(X_train.shape)
# class distribution

X_train['label'].unique()
import matplotlib.pyplot as plt



# how are the grey values distributed?

d = X_train.describe().T



ax = d.plot(y=['mean'], figsize=(20, 5), color='Black', kind='line') # mean values in black

plt.fill_between(x=d.index, y1='min', y2='max', data=d, alpha=.3) # min/max values filled in with blue

plt.show()
# train/ validation samples

msk = np.random.rand(len(X_train)) < 0.8

X_valid = X_train[~msk]

X_train = X_train[msk]

print ("{} samples for train, {} for validation".format(len(X_train), len(X_valid)))
# class distribution

X_train['label'].value_counts().plot(kind='bar')
# network specific parameters (layers' input/output dimensions)

n_inputs = 28*28 # input size

n_hidden1 = 300 # this is the output size of hidden layer 1

n_hidden2 = 100 # this is the output size of hidden layer 2

n_outputs = 10 # this is the output (final) layer size (will output the probability of each class from the 10 classes of digits)
import tensorflow as tf



# i'm gonna put the gloabl constant definitions here/ To be used later in the training step

n_iterations = 50

n_batches = 33 # nb batches (subsets used for iterative training), ie each batch is approx. 1000 size

learn_rate = 0.00003 # learning rate parameter (fixed empirically. Other values might be tested)
# place holders for the very first (input) layer

X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='X') # None means not-specified (varying) size

y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
# a generic function used to create the different layers of the model

def create_layer(input_layer, n_neurons, layer_name="", activation_fun=None):

    with tf.name_scope(layer_name):

        n_inputs = int(input_layer.get_shape()[1])

        initial_value = tf.truncated_normal((n_inputs, n_neurons)) # initial value (will updated at each iteration)

        w = tf.Variable(initial_value, name="weight") # weight vector, initiazed to initial_value

        b = tf.Variable(tf.zeros([n_neurons]), name="bias") # bias vector

        op = tf.matmul(input_layer, w) + b

        if activation_fun:

            op = activation_fun(op)

        return op
# the overall model architecture

with tf.name_scope("myfnn"):

    h1 = create_layer(X, n_hidden1, layer_name='hl1', activation_fun=tf.nn.relu)

    h2 = create_layer(h1, n_hidden2, layer_name='hl2', activation_fun=tf.nn.relu)

    logits = create_layer(h2, n_outputs, layer_name='output') # output layer with no activation function
# cost function

with tf.name_scope('loss'):

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

    loss = tf.reduce_mean(entropy)



#  train operation

with tf.name_scope("train"):

    optimizer = tf.train.GradientDescentOptimizer(learn_rate)

    training_op = optimizer.minimize(loss)
#  eval node

with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1) # predicted class is the most probable one

    acc = tf.reduce_mean(tf.cast(correct, tf.float32)) # a simple accuray measure
# result container

res = pd.DataFrame({'epoch':[], 'loss':[], 'acc_train':[], 'acc_valid':[]})



# we need to know the exact batch size to split data into batches

batch_size = int(len(X_train)/n_batches)

print ('batch size (auto computed) = {}'.format(str(batch_size)))

print ('n_iterations to go = {}'.format(str(n_iterations)))



# this operation is important before starting any training (initialize all variables with the value specified in definition)

init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    print ('it\tloss_t\tacc_t\tacc_v')

    

    for iteration_id in range(n_iterations):

        for batch_id in range(n_batches):

            i = batch_id * batch_size # start/end of current batch

            j = (batch_id + 1) * batch_size

            xx_train, yy_train = X_train.iloc[i:j, 1:], X_train.iloc[i:j, 0]

            

            # train and update parameters from the current batch data

            sess.run(training_op, feed_dict={X:xx_train, y:yy_train})

            

        # perform an evaluation at the end of each iteration

        loss_val = sess.run([loss], feed_dict={X:X_train.iloc[:, 1:], y:X_train.iloc[:, 0]})

        # evaluate accuracy on train /eval data may reveal any overfitting effect

        acc_train_val = sess.run([acc], feed_dict={X:X_train.iloc[:, 1:], y:X_train.iloc[:, 0]})

        acc_valid_val = sess.run([acc], feed_dict={X:X_valid.iloc[:,1:], y:X_valid.iloc[:,0]})

        

        res = res.append({'epoch':iteration_id, 'loss':loss_val[0], 'acc_train':acc_train_val[0], 

                          'acc_valid':acc_valid_val[0]}, ignore_index=True)

        if iteration_id % 10 == 0:

            print('{}\t{}\t{}\t{}'.format(iteration_id, str(round(loss_val[0], 1)), 

                                          str(round(acc_train_val[0], 3)), str(round(acc_valid_val[0], 3))))
# plot loss values

res = res.set_index(['epoch'])

plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

plt.plot(res.pop('loss'))

plt.subplot(1, 2, 2)



plt.plot(res)

plt.legend(res.columns)