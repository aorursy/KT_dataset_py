# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from keras.utils import np_utils

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')

test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
X_train = np.array(train.iloc[:,1:]) # (60000,784)

y_train = np.array(train['label'])   # (60000,)



X_test = np.array(test.iloc[:,1:]) # (10000,784)

y_test = np.array(test['label']) # (10000,)
digit_1 = X_train[0].reshape(28,28)

plt.imshow(digit_1)

plt.show()
digit_1 = X_train[25].reshape(28,28)

plt.imshow(digit_1)

plt.show()
y_train = np_utils.to_categorical(y_train,10)

y_test = np_utils.to_categorical(y_test,10)
n_input   = 784 # input layer (28x28 pixels) 

n_hidden1 = 512 # 1st hidden layer

n_hidden2 = 256 # 2nd hidden layer

n_hidden3 = 128 # 3rd hidden layer

n_output  = 10  # output layer (0-9)
learning_rate = 1e-4

n_iterations = 1000

batch_size = 128

dropout = 0.5
X = tf.placeholder("float", [None, n_input])

Y = tf.placeholder("float", [None, n_output])

keep_prob = tf.placeholder(tf.float32)
weights = {

    'w1' : tf.Variable(tf.truncated_normal([n_input,n_hidden1], stddev=0.1)),

    'w2' : tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2], stddev=0.1)),

    'w3' : tf.Variable(tf.truncated_normal([n_hidden2,n_hidden3], stddev=0.1)),

    'out' : tf.Variable(tf.truncated_normal([n_hidden3,n_output], stddev=0.1)), 

}



biases = {

    'b1' : tf.Variable(tf.constant(0.1, shape=[n_hidden1])),

    'b2' : tf.Variable(tf.constant(0.1, shape=[n_hidden2])),

    'b3' : tf.Variable(tf.constant(0.1, shape=[n_hidden3])),

    'out' : tf.Variable(tf.constant(0.1, shape=[n_output])),

}
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])

layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])

layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

layer_drop = tf.nn.dropout(layer_3, keep_prob)

output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



correct_pred = tf.equal(tf.argmax(output_layer,1), tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
for i in range(n_iterations):



    sess.run(train_step, feed_dict={

        X: X_train, Y: y_train, keep_prob: dropout

    })

    

    #print loss and accuracy (per minibatch)

    if i%100 == 0:

        minibatch_loss, minibatch_accuracy = sess.run(

            [cross_entropy, accuracy],

            feed_dict={X: X_train, Y: y_train, keep_prob: 1.0}

        )

        print("Iteration", str(i),"\t| Loss=",str(minibatch_loss),"\t| Accuracy=",str(minibatch_accuracy))
test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0})

print("\nAccuracy on test set:",test_accuracy)