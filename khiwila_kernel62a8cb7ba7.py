import math

import numpy as np

import pandas as pd

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

from keras.models import Sequential





%matplotlib inline

def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



X_train = np.asarray(train_data.drop(['label'], axis = 1))

y_train = np.asarray(train_data[['label']])

train_set_y_orig = y_train.reshape((1, y_train.shape[0]))



X_test = np.asarray(test_data)



# Loading the data (signs)

X_train_orig=X_train

Y_train_orig=train_set_y_orig 



train_set_y_orig = y_train.reshape((1, y_train.shape[0]))



# Loading the data (signs)

X_train_orig=X_train

Y_train_orig=train_set_y_orig 

X_test_orig=X_test



print(X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape )
#Normalize data

X_train = X_train_orig/float(255)

X_test = X_test_orig/float(255)

Y_train = convert_to_one_hot(Y_train_orig, 10).T

#Y_test = convert_to_one_hot(Y_test_orig, 10).T

print ("number of training examples = " + str(X_train.shape[0]))

#print ("number of test examples = " + str(X_test.shape[0]))

#Reshape data

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

#print ("Y_test shape: " + str(Y_test.shape))

conv_layers = {}
def create_placeholders(n_H0, n_W0, n_C0, n_y):

    """

    Creates the placeholders for the tensorflow session.

    

    Arguments:

    n_H0 -- scalar, height of an input image

    n_W0 -- scalar, width of an input image

    n_C0 -- scalar, number of channels of the input

    n_y -- scalar, number of classes

        

    Returns:

    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"

    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"

    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0)) 

    Y = tf.placeholder(tf.float32, shape=(None, n_y))



    return X, Y
def initialize_parameters():

    """

    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

                        W1 : [f, f, n_C_filter_prev, n_C]

                        W2 : [f, f, n_C_filter_prev, n_C]

    Returns:

    parameters -- a dictionary of tensors containing W1, W2

    """

    

    tf.set_random_seed(1)                              # so that your "random" numbers match ours



    W1 = tf.get_variable("W1", [2, 2, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))



    parameters = {"W1": W1,

                  "W2": W2}

    

    return parameters
def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "W2"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    W2 = parameters['W2']

    

    # CONV2D: stride of S, padding 'SAME'

    Z1 = tf.nn.conv2d(X,W1, strides = [1,2,2,1], padding = 'SAME')

    # RELU

    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window axa, stride S, padding 'SAME'

    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    # CONV2D: filters W2, stride S, padding 'SAME'

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,2,2,1], padding = 'SAME')

    # RELU

    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window fxf, stride S, padding 'SAME'

    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    # FLATTEN

    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).

    # N neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 

    Z3 = tf.contrib.layers.fully_connected(P2, 10,activation_fn=None)

    

    return Z3
def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)

    Y -- "true" labels vector placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    

    return cost
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)

    mini_batch_size - size of the mini-batches, integer

    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    m = X.shape[0]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X_train, Y_train, X_test, Y_test, X_test_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):



    """

    Implements a three-layer ConvNet in Tensorflow:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X_train -- training set

    Y_train -- test set

    X_test -- training set

    Y_test -- test set

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

    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)

    seed = 3                                          # to keep results consistent (numpy seed)

    (m, n_H0, n_W0, n_C0) = X_train.shape             

    n_y = Y_train.shape[1]                            

    costs = []                                        # To keep track of the cost

    

    # Create Placeholders of the correct shape

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)



    # Inilitialize parameters 

    parameters = initialize_parameters()

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters)

    

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z3, Y)

    

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    # Initialize all the variables globally

    init = tf.global_variables_initializer()

     

    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        

        # Do the training loop

        for epoch in range(num_epochs):



            minibatch_cost = 0.

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:



                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).

                _ , temp_cost = sess.run([optimizer,cost], feed_dict ={X:minibatch_X, Y:minibatch_Y})

                

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

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train set:", X_train.shape[0])         

        print("Train Accuracy:", train_accuracy)

        print("Test set:", X_test.shape[0]) 

        print("Test Accuracy:", test_accuracy)

        

        # Calculate Prediction in test_test set

        prediction=predict_op.eval({X: X_test_test})

        submission = pd.DataFrame({'ImageId':range(1,len(X_test_test)+1),'Label':prediction})

        submission.to_csv("submission_set.csv", index=False)

        print("Prediction_test_test set:", prediction.shape[0])

        print("Prediction_test_test_set:", prediction)

        print(submission.head())

        

        sess.close()

        return train_accuracy,test_accuracy, parameters
inicio_del_train=0

numero_samples=35000

numero_test_samples=5000





inicio_del_test=numero_samples+1



A_train=X_train[inicio_del_train:(inicio_del_train+numero_samples)]

Ytrain=Y_train[inicio_del_train:(inicio_del_train+numero_samples)]



A_train_test=X_train[inicio_del_test:(inicio_del_test+numero_test_samples)]

Ytrain_test=Y_train[inicio_del_test:(inicio_del_test+numero_test_samples)]



_,_, parameters = model(A_train, Ytrain, A_train_test, Ytrain_test, X_test)


