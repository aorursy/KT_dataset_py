# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from sklearn.preprocessing import StandardScaler



import os

import time



from __future__ import absolute_import, division, print_function

import numbers

from tensorflow.contrib import layers

from tensorflow.python.framework import ops

from tensorflow.python.framework import tensor_shape

from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import math_ops

from tensorflow.python.ops import random_ops

from tensorflow.python.ops import array_ops

from tensorflow.python.layers import utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/fashion-mnist_train.csv")

df_test = pd.read_csv("../input/fashion-mnist_test.csv")



train_x = df_train[df_train.columns[1:]]

train_y = df_train["label"]



test_x = df_test[df_test.columns[1:]]

test_y = df_test["label"]



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
def dense_to_one_hot(labels_dense, num_classes=10):

        num_labels = labels_dense.shape[0]

        index_offset = np.arange(num_labels) * num_classes

        labels_one_hot = np.zeros((num_labels, num_classes))

        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        

        return pd.DataFrame(labels_one_hot)
train_y = dense_to_one_hot(train_y, num_classes=10)

test_y = dense_to_one_hot(test_y, num_classes=10)



print(train_y.shape)

print(test_y.shape)
# Parameters

learning_rate = 0.05

training_epochs = 50

batch_size = 100

display_step = 1



# Network Parameters

n_hidden_1 = 784 # 1st layer number of features

n_hidden_2 = 784 # 2nd layer number of features

n_input = 784 # FMNIST data input (img shape: 28*28)

n_classes = 10 # FMNIST total classes (0-9 digits)



# tf Graph input

x = tf.placeholder("float", [None, n_input])

y = tf.placeholder("float", [None, n_classes])

dropoutRate = tf.placeholder(tf.float32)

is_training= tf.placeholder(tf.bool)
def selu(x):

    with ops.name_scope('selu') as scope:

        alpha = 1.6732632423543772848170429916717

        scale = 1.0507009873554804934193349852946

        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 

                 noise_shape=None, seed=None, name=None, training=False):

    """Dropout to a value with rescaling."""



    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):

        keep_prob = 1.0 - rate

        x = ops.convert_to_tensor(x, name="x")

        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:

            raise ValueError("keep_prob must be a scalar tensor or a float in the "

                                             "range (0, 1], got %g" % keep_prob)

        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")

        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())



        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")

        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())



        if tensor_util.constant_value(keep_prob) == 1:

            return x



        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)

        random_tensor = keep_prob

        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)

        binary_tensor = math_ops.floor(random_tensor)

        ret = x * binary_tensor + alpha * (1-binary_tensor)



        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))



        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)

        ret = a * ret + b

        ret.set_shape(x.get_shape())

        return ret



    with ops.name_scope(name, "dropout", [x]) as name:

        return utils.smart_cond(training,

            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),

            lambda: array_ops.identity(x))

# (1) Scale input to zero mean and unit variance

scaler = StandardScaler().fit(train_x)

test_x = scaler.transform(test_x)
# Create model

def multilayer_perceptron(x, weights, biases, rate, is_training):

    # Hidden layer with SELU activation

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    layer_1 = selu(layer_1)

    layer_1 = dropout_selu(layer_1,rate, training=is_training)

    

    # Hidden layer with SELU activation

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    layer_2 = selu(layer_2)

    layer_2 = dropout_selu(layer_2,rate, training=is_training)



    # Output layer with linear activation

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer
# Store layers weight & bias

weights = {

    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=np.sqrt(1/n_input))),

    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=np.sqrt(1/n_hidden_1))),

    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes],stddev=np.sqrt(1/n_hidden_2)))

}

biases = {

    'b1': tf.Variable(tf.random_normal([n_hidden_1],stddev=0)),

    'b2': tf.Variable(tf.random_normal([n_hidden_2],stddev=0)),

    'out': tf.Variable(tf.random_normal([n_classes],stddev=0))

}

# Construct model

pred = multilayer_perceptron(x, weights, biases, rate=dropoutRate, is_training=is_training)



# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)



 # Test model

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

         

# Initializing the variables

init = tf.global_variables_initializer()
# Launch the graph

gpu_options = tf.GPUOptions(allow_growth=True)



with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(init)



    start_time = time.time()

    

    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        total_batch = int(train_x.shape[0]/batch_size)

        # Loop over all batches

        for i in range(total_batch):

            randidx = np.random.randint(len(train_x), size=batch_size)

            batch_xs = train_x.iloc[randidx]

            batch_ys = train_y.iloc[randidx]

            batch_xs = scaler.transform(batch_xs)

            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,

                                                          y: batch_ys, dropoutRate: 0.05, is_training:True})



            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if epoch % display_step == 0:

            print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))

            

            accTrain, costTrain = sess.run([accuracy, cost], 

                                                        feed_dict={x: batch_xs, y: batch_ys, 

                                                                   dropoutRate: 0.0, is_training:False})

            

            print("Train-Accuracy:", accTrain,"Train-Loss:", costTrain)



            accTest, costVal = sess.run([accuracy, cost], 

                                                     feed_dict={x: test_x, y: test_y, 

                                                                   dropoutRate: 0.0, is_training:False})



            print("Test-Accuracy:", accTest,"\n")

            

    print("--- %s seconds ---" % (time.time() - start_time))