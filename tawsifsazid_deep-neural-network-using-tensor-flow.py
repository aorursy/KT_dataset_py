#Author: Muhammad Tawsif Sazid



import os

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import sklearn.datasets

import sklearn.linear_model

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.python.framework import ops





def create_placeholders(n_x, n_y):

    

    #print(n_x)

    #print(n_y)

    X = tf.placeholder(tf.float32, [n_x, None], name= "X")

    Y = tf.placeholder(tf.float32, [n_y, None], name ="Y")

    

    print ("X = " + str(X))

    print ("Y = " + str(Y))

    return X, Y



def initialize_parameters():

    

    W1 = tf.get_variable("W1", [25, 4], initializer = tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [2, 12], initializer = tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable("b3", [2, 1], initializer = tf.zeros_initializer())

    

    

    print(W1)

    parameters ={

        "W1" : W1,

        "b1" : b1,

        "W2" : W2,

        "b2" : b2,

        "W3" : W3,

        "b3" : b3

    }

    

    return parameters



def forward_propagation(X, parameters):

    

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    W3 = parameters["W3"]

    b3 = parameters["b3"]

    

    Z1 = tf.add(tf.matmul(W1,X), b1)

    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(W2,A1), b2)

    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(W3,A2), b3)

    

    return Z3



def compute_cost(Z3, Y):

    

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    

    return cost



def random_mini_batches(X, Y, mini_batch_size, seed):

    

    

    

    #np.random.seed(seed)            

    m = X.shape[1]                 

    mini_batches = []

        

   

    permutation = list(np.random.permutation(m))

    #print("asasdsad" + str(permutation))



    shuffled_X = X[:, permutation]

    #print(shuffled_X.shape)

    #print("sadasdsad" + str(Y.shape))

    shuffled_Y = Y[:, permutation].reshape((2,m))

    #print(shuffled_Y.shape)

    #shuffled_Y = Y[:, (int)(permutation)]

  

    num_complete_minibatches = math.floor(m/mini_batch_size)

    #num_complete_minibatches = (int)(m/mini_batch_size) 

    for k in range(0, num_complete_minibatches):

     

        mini_batch_X = shuffled_X[:, mini_batch_size*k : mini_batch_size*(k+1)]

        mini_batch_Y = shuffled_Y[:, mini_batch_size*k : mini_batch_size*(k+1)]

       

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    

    if m % mini_batch_size != 0:

        

        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size :]

        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size :]

    

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches



def model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, mini_batch_size, print_cost = True):

    #print(X_train.shape)

    #print(Y_train.shape)

    #print(X_test.shape)

    #print(Y_test.shape)

    ops.reset_default_graph()

    (n_x, m) = X_train.shape

    seed = 3

    #print(nx)

    #print(m)

    n_y = Y_train.shape[0]

    #print(n_y)

    costs = []

    #print(n_x)

    #print(n_y)

    

    X, Y = create_placeholders(n_x, n_y)

        

    parameters = initialize_parameters()

        

    Z3 = forward_propagation(X, parameters)

    

    cost = compute_cost(Z3, Y)

    print(str(cost))

    

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        

        sess.run(init)

        

        for epoch in range(num_epochs):

            

            epoch_cost = 0

            

            num_mini_batches = (int)(m / mini_batch_size)

            #print(str(num_mini_batches))

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)

            

            for minibatch in minibatches:

                

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer,cost] , feed_dict ={X:minibatch_X, Y:minibatch_Y})



                epoch_cost += minibatch_cost/num_mini_batches



            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)



        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()

        

        parameters = sess.run(parameters)

        

        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    

    

def data_input_and_process_function():

    

    a = np.loadtxt("/kaggle/input/banknote-detection-authentication/data_banknote_authentication.txt",delimiter=',')

    #print(a.shape)

    #print(a)

     

    #checking nan values

    #print(np.isnan(a).any())

     

    y = np.array(a[:,4]).reshape(1372,1)

    #print(y.shape)

    #print(y)

     

    a = np.delete(a, 4, 1)

    #print(a.shape)

    #print(a)

     

    train_set_X, test_set_X, train_set_Y, test_set_Y = train_test_split(a, y, test_size=0.3)

    #print(train_set_X.shape)

    #print(test_set_X.shape)

    #print(train_set_Y.shape)

    #print(test_set_Y.shape)

    ''''

    train_set_X = train_set_X.reshape(4, 960)

    test_set_X = test_set_X.reshape(4, 412)

    train_set_Y = train_set_Y.reshape(1, 960)

    test_set_Y = test_set_Y.reshape(1, 412)

     

    #print(train_set_X.shape)

    #print(test_set_X.shape)

    #print(train_set_Y.shape)

    #print(test_set_Y.shape)

     

    #Normalize

    train_set_X = train_set_X - np.mean(train_set_X)

    train_set_X = train_set_X / np.std(train_set_X)

     

    test_set_X = test_set_X - np.mean(test_set_X)

    test_set_X = test_set_X / np.std(test_set_X)

    '''  

     # Flatten the training and test images

     # Flatten the training and test images

    X_train_flatten = train_set_X.reshape(train_set_X.shape[0], -1).T

    X_test_flatten = test_set_X.reshape(test_set_X.shape[0], -1).T

    # Normalize image vectors

    #X_train = X_train_flatten/255.

    #X_test = X_test_flatten/255.

    X_train = X_train_flatten - np.mean(X_train_flatten)

    X_train = X_train_flatten / np.std(X_train_flatten)

    

    X_test = X_test_flatten - np.mean(X_test_flatten)

    X_test = X_test_flatten / np.std(X_test_flatten)

    # Convert training and test labels to one hot matrices

   



    train_set_Y = train_set_Y.reshape(1, 960).astype(int)

    test_set_Y = test_set_Y.reshape(1, 412).astype(int)

    

    #Y_train = tf.one_hot(train_set_Y, depth = 2, axis = 0)

    #Y_test = tf.one_hot(test_set_Y, depth = 2, axis = 0)

    

    ############## one hot encoding ############################

    #print(train_set_Y)

    classes = 2

    targets = np.array(train_set_Y).reshape(-1)

    Y_train = np.eye(classes)[targets]

    Y_train = Y_train.transpose(1,0)

    #print(Y_train)

    

    targets = np.array(test_set_Y).reshape(-1)

    Y_test = np.eye(classes)[targets]

    Y_test = Y_test.transpose(1,0)

    #print(Y_train)

    #sess = tf.Session()

    #one_hot = sess.run(Y_train)

    #sess.close()



    print ("number of training examples = " + str(X_train.shape[1]))

    print ("number of test examples = " + str(X_test.shape[1]))

    print ("X_train shape: " + str(X_train.shape))

    print ("Y_train shape: " + str(Y_train.shape))

    print ("X_test shape: " + str(X_test.shape))

    print ("Y_test shape: " + str(Y_test.shape))    

    

    return X_train, X_test, Y_train, Y_test



def main_func():

    

    print("Hello World in Deep Learning")

    X_train, X_test, Y_train, Y_test = data_input_and_process_function()

    parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, mini_batch_size = 64)

    



main_func()

    



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory









# Any results you write to the current directory are saved as output.

