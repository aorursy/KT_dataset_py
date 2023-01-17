import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops



def load_dataset():

    train_dataset = h5py.File('/kaggle/input/train_signs.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('/kaggle/input/test_signs.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes





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





def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y





def forward_propagation_for_predict(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

                  the shapes are given in initialize_parameters

    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3'] 

                                                           # Numpy Equivalents:

    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2

    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    

    return Z3



def predict(X, parameters):

    

    W1 = tf.convert_to_tensor(parameters["W1"])

    b1 = tf.convert_to_tensor(parameters["b1"])

    W2 = tf.convert_to_tensor(parameters["W2"])

    b2 = tf.convert_to_tensor(parameters["b2"])

    W3 = tf.convert_to_tensor(parameters["W3"])

    b3 = tf.convert_to_tensor(parameters["b3"])

    

    params = {"W1": W1,

              "b1": b1,

              "W2": W2,

              "b2": b2,

              "W3": W3,

              "b3": b3}

    

    x = tf.placeholder("float", [12288, 1])

    

    z3 = forward_propagation_for_predict(x, params)

    p = tf.argmax(z3)

    

    sess = tf.Session()

    prediction = sess.run(p, feed_dict = {x: X})

        

    return prediction



def predict(X, parameters):

   

    W1 = tf.convert_to_tensor(parameters["W1"])

    b1 = tf.convert_to_tensor(parameters["b1"])

    W2 = tf.convert_to_tensor(parameters["W2"])

    b2 = tf.convert_to_tensor(parameters["b2"])

#    W3 = tf.convert_to_tensor(parameters["W3"])

#    b3 = tf.convert_to_tensor(parameters["b3"])

   

#    params = {"W1": W1,

#              "b1": b1,

#              "W2": W2,

#              "b2": b2,

#              "W3": W3,

#              "b3": b3}



    params = {"W1": W1,

             "b1": b1,

             "W2": W2,

             "b2": b2}    

   

    x = tf.placeholder("float", [12288, 1])

   

    z3 = forward_propagation(x, params)

    p = tf.argmax(z3)

   

    with tf.Session() as sess:

        prediction = sess.run(p, feed_dict = {x: X})

       

    return prediction
import scipy

from PIL import Image

from scipy import ndimage

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

# from cnn_utils import *



%matplotlib inline

np.random.seed(1)
# Loading the data (signs)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Example of a picture

index = 6

plt.imshow(X_train_orig[index])

print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
X_train = X_train_orig/255.

X_test = X_test_orig/255.

Y_train = convert_to_one_hot(Y_train_orig, 6).T

Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))

conv_layers = {}
# GRADED FUNCTION: create_placeholders



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



    ### START CODE HERE ### (≈2 lines)

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))

    Y = tf.placeholder(tf.float32, shape=(None, n_y))

    ### END CODE HERE ###

    

    return X, Y
X, Y = create_placeholders(64, 64, 3, 6)

print ("X = " + str(X))

print ("Y = " + str(Y))
# GRADED FUNCTION: initialize_parameters



def initialize_parameters():

    """

    Initializes weight parameters to build a neural network with tensorflow. The shapes are:

                        W1 : [4, 4, 3, 8]

                        W2 : [2, 2, 8, 16]

    Returns:

    parameters -- a dictionary of tensors containing W1, W2

    """

    

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

        

    ### START CODE HERE ### (approx. 2 lines of code)

    W1 = tf.get_variable("W1", [4,4,3,8], initializer= tf.contrib.layers.xavier_initializer(seed = 0))

    W2 = tf.get_variable("W2", [2,2,8,16], initializer= tf.contrib.layers.xavier_initializer(seed = 0))

    ### END CODE HERE ###



    parameters = {"W1": W1,

                  "W2": W2}

    

    return parameters
tf.reset_default_graph()

with tf.Session() as sess_test:

    parameters = initialize_parameters()

    init = tf.global_variables_initializer()

    sess_test.run(init)

    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))

    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
# GRADED FUNCTION: forward_propagation



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

    

    ### START CODE HERE ###

    # CONV2D: stride of 1, padding 'SAME'

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')

    # RELU

    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'

    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'

    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')    

    # RELU

    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'

    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    # FLATTEN

    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).

    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 

    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    ### END CODE HERE ###



    return Z3
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

    

    ### START CODE HERE ###

    # CONV2D: stride of 1, padding 'SAME'

    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')

    # RELU

    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'

    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')

    # RELU

    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'

    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    # FLATTEN

    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).

    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 

    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)

    ### END CODE HERE ###



    return Z3
tf.reset_default_graph()



with tf.Session() as sess:

    np.random.seed(1)

    X, Y = create_placeholders(64, 64, 3, 6)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})

    print("Z3 = " + str(a))
# GRADED FUNCTION: compute_cost 



def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)

    Y -- "true" labels vector placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    

    ### START CODE HERE ### (1 line of code)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    ### END CODE HERE ###

    

    return cost
tf.reset_default_graph()



with tf.Session() as sess:

    np.random.seed(1)

    X, Y = create_placeholders(64, 64, 3, 6)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})

    print("cost = " + str(a))
# GRADED FUNCTION: model



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,

          num_epochs = 100, minibatch_size = 64, print_cost = True):

    """

    Implements a three-layer ConvNet in Tensorflow:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    

    Arguments:

    X_train -- training set, of shape (None, 64, 64, 3)

    Y_train -- test set, of shape (None, n_y = 6)

    X_test -- training set, of shape (None, 64, 64, 3)

    Y_test -- test set, of shape (None, n_y = 6)

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

    ### START CODE HERE ### (1 line)

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    ### END CODE HERE ###



    # Initialize parameters

    ### START CODE HERE ### (1 line)

    parameters = initialize_parameters()

    ### END CODE HERE ###

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    ### START CODE HERE ### (1 line)

    Z3 = forward_propagation(X, parameters)

    ### END CODE HERE ###

    

    # Cost function: Add cost function to tensorflow graph

    ### START CODE HERE ### (1 line)

    cost = compute_cost(Z3, Y)

    ### END CODE HERE ###

    

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.

    ### START CODE HERE ### (1 line)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    ### END CODE HERE ###

    

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

                ### START CODE HERE ### (1 line)

                _ , temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})

                ### END CODE HERE ###

                

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

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

                

        return train_accuracy, test_accuracy, parameters
_, _, parameters = model(X_train, Y_train, X_test, Y_test)