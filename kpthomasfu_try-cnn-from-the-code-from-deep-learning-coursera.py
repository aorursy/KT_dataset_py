import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



import tensorflow as tf

from tensorflow.python.framework import ops



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X_train = train.iloc[:40000, 1:].values

Y_train = train.iloc[:40000, 0].values

X_val = train.iloc[40000:, 1:].values

Y_val = train.iloc[40000:, 0].values



X_test = test.iloc[:].values
# Reshape X to (28,28)



X_train_res = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))

X_val_res = np.reshape(X_val, (X_val.shape[0], 28, 28, 1))

X_test_res = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
# Create one_hot matrix to Y



def convert_one_hot(Y):

    

    Y_one = tf.one_hot(Y, depth=10, axis=-1)

    

    return Y_one
Y_train_1h = convert_one_hot(Y_train)

Y_val_1h = convert_one_hot(Y_val)



with tf.Session() as sess:

    Y_train_one = sess.run(Y_train_1h)

    Y_val_one = sess.run(Y_val_1h)
print("X_train_res shape " + str(X_train_res.shape))

print("X_val_res shape " + str(X_val_res.shape))

print("X_test_res shape " + str(X_test_res.shape))

print("Y_train_one shape " + str(Y_train_one.shape))

print("Y_val_one shape " + str(Y_val_one.shape))
# Create placeholder for X, Y



def create_placeholders(n_W, n_H, n_C, n_Y):

    # n_W: Width of the image

    # n_H: Height of the image

    # n_C: Layer of the image

    # n_Y: Number of class

    

    X = tf.placeholder(dtype=tf.float32, shape=[None, n_W, n_H, n_C])

    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_Y])

    

    return X, Y
# Initialize weight parameter



def initialize_parameters():

    

    W1 = tf.get_variable("W1", [8, 8, 1, 8], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

    W2 = tf.get_variable("W2", [2, 2, 8, 16], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())



    parameters = {"W1": W1,

                  "W2": W2}

    

    return parameters
# Forward propagation



def forward_propagation(X, parameters):



    #Implements the forward propagation for the model:

    #CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    W1 = parameters['W1']

    W2 = parameters['W2']

    

    # CONV2D: stride of 1, padding 'SAME'

    Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')

    # RELU

    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'

    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'

    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')

    # RELU

    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'

    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')

    # FLATTEN

    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).

    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 

    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn = None)



    return Z3
def compute_cost(Z3, Y):



    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    

    return cost
def model(X_train, Y_train, X_val, Y_val, learning_rate = 0.009, num_epochs = 101, minibatch_size = 128, print_cost = True):

    """

    Implements a three-layer ConvNet in Tensorflow:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X_train -- training set, of shape (None, 64, 64, 1)

    Y_train -- test set, of shape (None, n_y = 10)

    X_test -- training set, of shape (None, 64, 64, 1)

    Y_test -- test set, of shape (None, n_y = 10)

    learning_rate -- learning rate of the optimization

    num_epochs -- number of epochs of the optimization loop

    minibatch_size -- size of a minibatch

    print_cost -- True to print the cost every 100 epochs

    

    Returns:

    train_accuracy -- real number, accuracy on the train set (X_train)

    test_accuracy -- real number, testing accuracy on the test set (X_test)

    parameters -- parameters learnt by the model. They can then be used to predict.

    """  

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables 

    seed = 0

    (m, n_H, n_W, n_C) = X_train.shape             

    n_y = Y_train.shape[1]                            

    costs = []                                        # To keep track of the cost

    

    # Create Placeholders of the correct shape

    X, Y = create_placeholders(n_H, n_W, n_C, n_y)



    # Initialize parameters

    parameters = initialize_parameters()

    

    # Forward propagation

    Z3 = forward_propagation(X, parameters)

    

    # Cost function

    cost = compute_cost(Z3, Y)

    

    # Backpropagation

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        sess.run(init)

        

        for epoch in range(num_epochs):



            minibatch_cost = 0.

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed += 1



            for minibatch in range(num_minibatches):



                minibatch_X = X_train[minibatch*minibatch_size:(minibatch+1)*minibatch_size, :, :, :]

                minibatch_Y = Y_train[minibatch*minibatch_size:(minibatch+1)*minibatch_size, :]                

                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})                

                minibatch_cost += temp_cost / num_minibatches

                



            # Print the cost every epoch

            if print_cost == True and epoch % 5 == 0:

                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0:

                costs.append(minibatch_cost)

        

        

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # Calculate the correct predictions

        predict_op = tf.argmax(Z3, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_val, Y: Y_val})

        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

                

        return train_accuracy, test_accuracy, parameters
_, _, parameters = model(X_train_res, Y_train_one, X_val_res, Y_val_one)