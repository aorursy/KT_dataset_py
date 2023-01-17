# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

import scipy

from tensorflow.python.framework import ops

import random

import scipy

import os



from scipy import ndimage







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head(5)
test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = ((train.ix[:,1:]).values).astype('float32')

Y_train = ((train.ix[:,:1]).values).astype('int32')

X_test = test.values.astype('float32')
X_train.shape
Y_train.shape
X_test.shape
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0],28,28,1)

X_test.shape
index = 3

plt.imshow(X_train[index] [:,:,0])

plt.title(Y_train[index])
index = 4

plt.imshow(X_test[index][:,:,0])
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test =  X_test.reshape(X_test.shape[0],28,28,1)

X_test.shape
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

label_binarizer.fit(range(10))



def one_hot_encode(x):

    return label_binarizer.transform(x)
Y_train = one_hot_encode(x = Y_train)
X_train = X_train / 255

X_test  = X_test / 255
def create_placeholders(n_H0,n_W0,n_C0,n_Y):



    X = tf.placeholder(shape = (None,n_H0, n_W0,n_C0), dtype = tf.float32)

    Y = tf.placeholder(shape = (None,n_Y), dtype = tf.float32)

    return X,Y
def initialize_parameters():

    

    tf.set_random_seed(1)

    

    W1 = tf.Variable(tf.truncated_normal([4,4,1,8]))

    W2 = tf.Variable(tf.truncated_normal([2,2,8,16]))

    parameters = {"W1" : W1,

                  "W2" : W2}

    return parameters
def forward_propagation(X, parameters):

    W1 = parameters['W1']

    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize = [1 ,4,4,1], strides = [1,2,2,1] ,padding = 'SAME')

    Z2 = tf.nn.conv2d(P1 ,W2 ,strides = [1,1,1,1], padding = 'SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2 , num_outputs = 10 , activation_fn = None)

    return(Z3)
def compute_cost(Z3 , Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3 , labels = Y ))

    return cost
learning_rate = 0.009

num_epochs = 100

print_cost = True



                                     

(m, n_H0, n_W0, n_C0) = X_train.shape 

n_y = Y_train.shape[1]                            

costs = []                                        



X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)



parameters = initialize_parameters()



Z3 = forward_propagation(X, parameters)

predict = tf.argmax(Z3,1)



cost = compute_cost(Z3, Y)



optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)



init = tf.global_variables_initializer()



sess = tf.InteractiveSession()        

sess.run(init)

        

for epoch in range(num_epochs):

    sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})

    loss = sess.run(cost, feed_dict={X: X_train, Y: Y_train})

    if print_cost == True and epoch % 5 == 0:

        print ("Cost after epoch %i: %f" % (epoch, loss))

        if print_cost == True and epoch % 1 == 0:

            costs.append(loss)

        

        

plt.plot(np.squeeze(costs))

plt.ylabel('cost')

plt.xlabel('iterations (per tens)')

plt.title("Learning rate =" + str(learning_rate))

plt.show()



predict_op = tf.argmax(Z3, 1)

correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

predicted_labels = np.zeros((X_test.shape[0]))

predicted_labels = predict_op.eval(feed_dict = {X : X_test})

print(predicted_labels[0:5])



        

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy)

train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

print("Train Accuracy:", train_accuracy)
np.savetxt('Y_test.csv', 

           np.c_[range(1,len(X_test)+1),predicted_labels], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')