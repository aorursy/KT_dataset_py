# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

import h5py

import sklearn

import sklearn.datasets

import sklearn.linear_model

import scipy.io



def sigmoid(x):

    """

    Compute the sigmoid of x



    Arguments:

    x -- A scalar or numpy array of any size.



    Return:

    s -- sigmoid(x)

    """

    s = 1/(1+np.exp(-x))

    return s



def relu(x):

    """

    Compute the relu of x



    Arguments:

    x -- A scalar or numpy array of any size.



    Return:

    s -- relu(x)

    """

    s = np.maximum(0,x)

    

    return s



def load_planar_dataset(seed):

    

    np.random.seed(seed)

    

    m = 400 # number of examples

    N = int(m/2) # number of points per class

    D = 2 # dimensionality

    X = np.zeros((m,D)) # data matrix where each row is a single example

    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)

    a = 4 # maximum ray of the flower



    for j in range(2):

        ix = range(N*j,N*(j+1))

        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta

        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius

        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]

        Y[ix] = j

        

    X = X.T

    Y = Y.T



    return X, Y



def initialize_parameters(layer_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the dimensions of each layer in our network

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                    b1 -- bias vector of shape (layer_dims[l], 1)

                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])

                    bl -- bias vector of shape (1, layer_dims[l])

                    

    Tips:

    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1]. 

    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!

    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layer_dims) # number of layers in the network



    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters



def forward_propagation(X, parameters):

    """

    Implements the forward propagation (and computes the loss) presented in Figure 2.

    

    Arguments:

    X -- input dataset, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":

                    W1 -- weight matrix of shape ()

                    b1 -- bias vector of shape ()

                    W2 -- weight matrix of shape ()

                    b2 -- bias vector of shape ()

                    W3 -- weight matrix of shape ()

                    b3 -- bias vector of shape ()

    

    Returns:

    loss -- the loss function (vanilla logistic loss)

    """

        

    # retrieve parameters

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    W3 = parameters["W3"]

    b3 = parameters["b3"]

    

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    Z1 = np.dot(W1, X) + b1

    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2

    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3

    A3 = sigmoid(Z3)

    

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    

    return A3, cache



def backward_propagation(X, Y, cache):

    """

    Implement the backward propagation presented in figure 2.

    

    Arguments:

    X -- input dataset, of shape (input size, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)

    cache -- cache output from forward_propagation()

    

    Returns:

    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables

    """

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    

    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T)

    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    

    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    dW2 = 1./m * np.dot(dZ2, A1.T)

    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    

    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dW1 = 1./m * np.dot(dZ1, X.T)

    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,

                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,

                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    

    return gradients



def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters:

                    parameters['W' + str(i)] = Wi

                    parameters['b' + str(i)] = bi

    grads -- python dictionary containing your gradients for each parameters:

                    grads['dW' + str(i)] = dWi

                    grads['db' + str(i)] = dbi

    learning_rate -- the learning rate, scalar.

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

    """

    

    n = len(parameters) // 2 # number of layers in the neural networks



    # Update rule for each parameter

    for k in range(n):

        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]

        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]

        

    return parameters



def predict(X, y, parameters):

    """

    This function is used to predict the results of a  n-layer neural network.

    

    Arguments:

    X -- data set of examples you would like to label

    parameters -- parameters of the trained model

    

    Returns:

    p -- predictions for the given dataset X

    """

    

    m = X.shape[1]

    p = np.zeros((1,m), dtype = np.int)

    

    # Forward propagation

    a3, caches = forward_propagation(X, parameters)

    

    # convert probas to 0/1 predictions

    for i in range(0, a3.shape[1]):

        if a3[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0



    # print results



    #print ("predictions: " + str(p[0,:]))

    #print ("true labels: " + str(y[0,:]))

    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    

    return p



def compute_cost(a3, Y):

    """

    Implement the cost function

    

    Arguments:

    a3 -- post-activation, output of forward propagation

    Y -- "true" labels vector, same shape as a3

    

    Returns:

    cost - value of the cost function

    """

    m = Y.shape[1]

    

    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)

    cost = 1./m * np.nansum(logprobs)

    

    return cost



def load_dataset():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T

    test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    

    train_set_x = train_set_x_orig/255

    test_set_x = test_set_x_orig/255



    return train_set_x, train_set_y, test_set_x, test_set_y, classes





def predict_dec(parameters, X):

    """

    Used for plotting decision boundary.

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    X -- input data of size (m, K)

    

    Returns

    predictions -- vector of predictions of our model (red: 0 / blue: 1)

    """

    

    # Predict using forward propagation and a classification threshold of 0.5

    a3, cache = forward_propagation(X, parameters)

    predictions = (a3>0.5)

    return predictions



def load_planar_dataset(randomness, seed):

    

    np.random.seed(seed)

    

    m = 50

    N = int(m/2) # number of points per class

    D = 2 # dimensionality

    X = np.zeros((m,D)) # data matrix where each row is a single example

    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)

    a = 2 # maximum ray of the flower



    for j in range(2):

        

        ix = range(N*j,N*(j+1))

        if j == 0:

            t = np.linspace(j, 4*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta

            r = 0.3*np.square(t) + np.random.randn(N)*randomness # radius

        if j == 1:

            t = np.linspace(j, 2*3.1415*(j+1),N) #+ np.random.randn(N)*randomness # theta

            r = 0.2*np.square(t) + np.random.randn(N)*randomness # radius

            

        X[ix] = np.c_[r*np.cos(t), r*np.sin(t)]

        Y[ix] = j

        

    X = X.T

    Y = Y.T



    return X, Y



def plot_decision_boundary(model, X, y):

    # Set min and max values and give it some padding

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1

    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid

    Z = model(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

    plt.show()

    

def load_2D_dataset():

    data = scipy.io.loadmat('datasets/data.mat')

    train_X = data['X'].T

    train_Y = data['y'].T

    test_X = data['Xval'].T

    test_Y = data['yval'].T



    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    

    return train_X, train_Y, test_X, test_Y

def compute_loss(a3, Y):

    

    """

    Implement the loss function

    

    Arguments:

    a3 -- post-activation, output of forward propagation

    Y -- "true" labels vector, same shape as a3

    

    Returns:

    loss - value of the loss function

    """

    

    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)

    loss = 1./m * np.nansum(logprobs)

    

    return loss
data=pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
X=data.drop(columns=['target'],axis=1)

y=data['target'].values
X=np.array(X)
sc=StandardScaler()
X=sc.fit_transform(X)
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):

    """

    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    

    Arguments:

    X -- input data, of shape (2, number of examples)

    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)

    learning_rate -- learning rate for gradient descent 

    num_iterations -- number of iterations to run gradient descent

    print_cost -- if True, print the cost every 1000 iterations

    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    

    Returns:

    parameters -- parameters learnt by the model

    """

        

    grads = {}

    costs = [] # to keep track of the loss

    m = X.shape[1] # number of examples

    layers_dims = [X.shape[0], 10, 5, 1]

    

    # Initialize parameters dictionary.

    if initialization == "zeros":

        parameters = initialize_parameters_zeros(layers_dims)

    elif initialization == "random":

        parameters = initialize_parameters_random(layers_dims)

    elif initialization == "he":

        parameters = initialize_parameters_he(layers_dims)



    # Loop (gradient descent)



    for i in range(0, num_iterations):



        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.

        a3, cache = forward_propagation(X, parameters)

        

        # Loss

        cost = compute_loss(a3, Y)



        # Backward propagation.

        grads = backward_propagation(X, Y, cache)

        

        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

        

        # Print the loss every 1000 iterations

        if print_cost and i % 1000 == 0:

            print("Cost after iteration {}: {}".format(i, cost))

            costs.append(cost)

            

    # plot the loss

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
train_X,test_X,train_Y,test_Y=train_test_split( X, y, test_size=0.10, random_state=8)
train_X = train_X.T

train_Y = train_Y.reshape((1, train_Y.shape[0]))

test_X = test_X.T

test_Y = test_Y.reshape((1, test_Y.shape[0]))
# GRADED FUNCTION: initialize_parameters_zeros 



def initialize_parameters_zeros(layers_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the size of each layer.

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])

                    b1 -- bias vector of shape (layers_dims[1], 1)

                    ...

                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])

                    bL -- bias vector of shape (layers_dims[L], 1)

    """

    

    parameters = {}

    L = len(layers_dims)            # number of layers in the network

    

    for l in range(1, L):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###

    return parameters
parameters = initialize_parameters_zeros([13,10,1])

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
parameters = model(train_X, train_Y, initialization = "zeros")

print ("On the train set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# GRADED FUNCTION: initialize_parameters_random



def initialize_parameters_random(layers_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the size of each layer.

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])

                    b1 -- bias vector of shape (layers_dims[1], 1)

                    ...

                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])

                    bL -- bias vector of shape (layers_dims[L], 1)

    """

    

    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours

    parameters = {}

    L = len(layers_dims)            # integer representing the number of layers

    

    for l in range(1, L):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###



    return parameters

parameters = initialize_parameters_random([13, 10, 1])

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
parameters = model(train_X, train_Y, initialization = "random")

print ("On the train set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# GRADED FUNCTION: initialize_parameters_he



def initialize_parameters_he(layers_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the size of each layer.

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])

                    b1 -- bias vector of shape (layers_dims[1], 1)

                    ...

                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])

                    bL -- bias vector of shape (layers_dims[L], 1)

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layers_dims) - 1 # integer representing the number of layers

     

    for l in range(1, L + 1):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###

        

    return parameters
parameters = initialize_parameters_he([2, 4, 1])

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
parameters = model(train_X, train_Y, initialization = "he")

print ("On the train set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# 2nd part

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):

    """

    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    

    Arguments:

    X -- input data, of shape (input size, number of examples)

    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)

    learning_rate -- learning rate of the optimization

    num_iterations -- number of iterations of the optimization loop

    print_cost -- If True, print the cost every 10000 iterations

    lambd -- regularization hyperparameter, scalar

    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    

    Returns:

    parameters -- parameters learned by the model. They can then be used to predict.

    """

        

    grads = {}

    costs = []                            # to keep track of the cost

    m = X.shape[1]                        # number of examples

    layers_dims = [X.shape[0], 20, 3, 1]

    

    # Initialize parameters dictionary.

    parameters = initialize_parameters_he(layers_dims)



    # Loop (gradient descent)



    for i in range(0, num_iterations):



        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.

        if keep_prob == 1:

            a3, cache = forward_propagation(X, parameters)

        elif keep_prob < 1:

            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        

        # Cost function

        if lambd == 0:

            cost = compute_cost(a3, Y)

        else:

            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

            

        # Backward propagation.

        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 

                                            # but this assignment will only explore one at a time

        if lambd == 0 and keep_prob == 1:

            grads = backward_propagation(X, Y, cache)

        elif lambd != 0:

            grads = backward_propagation_with_regularization(X, Y, cache, lambd)

        elif keep_prob < 1:

            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        

        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

        

        # Print the loss every 10000 iterations

        if print_cost and i % 10000 == 0:

            print("Cost after iteration {}: {}".format(i, cost))

        if print_cost and i % 1000 == 0:

            costs.append(cost)

    

    # plot the cost

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('iterations (x1,000)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
parameters = model(train_X, train_Y)

print ("On the training set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# GRADED FUNCTION: compute_cost_with_regularization



def compute_cost_with_regularization(A3, Y, parameters, lambd):

    """

    Implement the cost function with L2 regularization. See formula (2) above.

    

    Arguments:

    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)

    Y -- "true" labels vector, of shape (output size, number of examples)

    parameters -- python dictionary containing parameters of the model

    

    Returns:

    cost - value of the regularized loss function (formula (2))

    """

    m = Y.shape[1]

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    W3 = parameters["W3"]

    

    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost

    

    ### START CODE HERE ### (approx. 1 line)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    ### END CO

    ### END CODER HERE ###

    

    cost = cross_entropy_cost + L2_regularization_cost

    

    return cost
A3,c=forward_propagation(train_X,parameters)


def backward_propagation_with_regularization(X, Y, cache, lambd):

    """

    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    

    Arguments:

    X -- input dataset, of shape (input size, number of examples)

    Y -- "true" labels vector, of shape (output size, number of examples)

    cache -- cache output from forward_propagation()

    lambd -- regularization hyperparameter, scalar

    

    Returns:

    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables

    """

    

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    

    dZ3 = A3 - Y

    

    ### START CODE HERE ### (approx. 1 line)

    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd * W3) / m

    ### END CODE HERE ###

    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    

    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    ### START CODE HERE ### (approx. 1 line)

    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m

    ### END CODE HERE ###

    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    

    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    ### START CODE HERE ### (approx. 1 line)

    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m

    ### END CODE HERE ###

    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,

                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 

                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    

    return gradients
grads = backward_propagation_with_regularization(train_X, train_Y, c, lambd = 0.7)

print ("dW1 = \n"+ str(grads["dW1"]))

print ("dW2 = \n"+ str(grads["dW2"]))

print ("dW3 = \n"+ str(grads["dW3"]))
parameters = model(train_X, train_Y, lambd = 0.7)

print ("On the train set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# GRADED FUNCTION: forward_propagation_with_dropout



def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):

    """

    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    

    Arguments:

    X -- input dataset, of shape (2, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":

                    W1 -- weight matrix of shape (20, 2)

                    b1 -- bias vector of shape (20, 1)

                    W2 -- weight matrix of shape (3, 20)

                    b2 -- bias vector of shape (3, 1)

                    W3 -- weight matrix of shape (1, 3)

                    b3 -- bias vector of shape (1, 1)

    keep_prob - probability of keeping a neuron active during drop-out, scalar

    

    Returns:

    A3 -- last activation value, output of the forward propagation, of shape (1,1)

    cache -- tuple, information stored for computing the backward propagation

    """

    

    np.random.seed(1)

    

    # retrieve parameters

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    W3 = parameters["W3"]

    b3 = parameters["b3"]

    

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    Z1 = np.dot(W1, X) + b1

    A1 = relu(Z1)

    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 

    D1 = np.random.rand(A1.shape[0], A1.shape[1])     # Step 1: initialize matrix D1 = np.random.rand(..., ...)

    D1 = D1 < keep_prob                            # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)

    A1 = A1 * D1                                      # Step 3: shut down some neurons of A1

    A1 = A1 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down

    ### END CODE HERE ###

    Z2 = np.dot(W2, A1) + b2

    A2 = relu(Z2)

    ### START CODE HERE ### (approx. 4 lines)

    D2 = np.random.rand(A2.shape[0], A2.shape[1])     # Step 1: initialize matrix D2 = np.random.rand(..., ...)

    D2 = D2 < keep_prob                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)                           

    A2 = A2 * D2                                      # Step 3: shut down some neurons of A2

    A2 = A2 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down

    ### END CODE HERE ###

    Z3 = np.dot(W3, A2) + b3

    A3 = sigmoid(Z3)

    

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    

    return A3, cache
A3, cache = forward_propagation_with_dropout(train_X, parameters, keep_prob = 0.7)

print ("A3 = " + str(A3))
# GRADED FUNCTION: backward_propagation_with_dropout



def backward_propagation_with_dropout(X, Y, cache, keep_prob):

    """

    Implements the backward propagation of our baseline model to which we added dropout.

    

    Arguments:

    X -- input dataset, of shape (2, number of examples)

    Y -- "true" labels vector, of shape (output size, number of examples)

    cache -- cache output from forward_propagation_with_dropout()

    keep_prob - probability of keeping a neuron active during drop-out, scalar

    

    Returns:

    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables

    """

    

    m = X.shape[1]

    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    

    dZ3 = A3 - Y

    dW3 = 1. / m * np.dot(dZ3, A2.T)

    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)

    ### START CODE HERE ### (≈ 2 lines of code)

    dA2 = dA2 * D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation

    dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    ### END CODE HERE ###

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    dW2 = 1. / m * np.dot(dZ2, A1.T)

    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    

    dA1 = np.dot(W2.T, dZ2)

    ### START CODE HERE ### (≈ 2 lines of code)

    dA1 = dA1 * D1              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation

    dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    ### END CODE HERE ###

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dW1 = 1. / m * np.dot(dZ1, X.T)

    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,

                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 

                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    

    return gradients
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)



print ("On the train set:")

predictions_train = predict(train_X, train_Y, parameters)

print ("On the test set:")

predictions_test = predict(test_X, test_Y, parameters)
# we see that the accuracy on the train set and test set came closer and the accuracy on the test score ncreased making our model away from the overfittin model it was earlier.

# GRADED FUNCTION: update_parameters_with_gd



def update_parameters_with_gd(parameters, grads, learning_rate):

    """

    Update parameters using one step of gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters to be updated:

                    parameters['W' + str(l)] = Wl

                    parameters['b' + str(l)] = bl

    grads -- python dictionary containing your gradients to update each parameters:

                    grads['dW' + str(l)] = dWl

                    grads['db' + str(l)] = dbl

    learning_rate -- the learning rate, scalar.

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

    """



    L = len(parameters) // 2 # number of layers in the neural networks



    # Update rule for each parameter

    for l in range(L):

        ### START CODE HERE ### (approx. 2 lines)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        ### END CODE HERE ###

        

    return parameters


parameters = update_parameters_with_gd(parameters, grads, learning_rate=0.01)

print("W1 =\n" + str(parameters["W1"]))

print("b1 =\n" + str(parameters["b1"]))

print("W2 =\n" + str(parameters["W2"]))

print("b2 =\n" + str(parameters["b2"]))
# mini batch gradient

# GRADED FUNCTION: random_mini_batches



def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples)

    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)

    mini_batch_size -- size of the mini-batches, integer

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    np.random.seed(seed)            # To make your "random" minibatches the same as ours

    m = X.shape[1]                  # number of training examples

    mini_batches = []

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((1,m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        ### START CODE HERE ### (approx. 2 lines)

        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]

        ### END CODE HERE ###

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        ### START CODE HERE ### (approx. 2 lines)

        end = m - mini_batch_size * math.floor(m / mini_batch_size)

        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]

        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]

        ### END CODE HERE ###

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
# momentum

# GRADED FUNCTION: initialize_velocity



def initialize_velocity(parameters):

    """

    Initializes the velocity as a python dictionary with:

                - keys: "dW1", "db1", ..., "dWL", "dbL" 

                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:

    parameters -- python dictionary containing your parameters.

                    parameters['W' + str(l)] = Wl

                    parameters['b' + str(l)] = bl

    

    Returns:

    v -- python dictionary containing the current velocity.

                    v['dW' + str(l)] = velocity of dWl

                    v['db' + str(l)] = velocity of dbl

    """

    

    L = len(parameters) // 2 # number of layers in the neural networks

    v = {}

    

    # Initialize velocity

    for l in range(L):

        ### START CODE HERE ### (approx. 2 lines)

        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])

        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])

        ### END CODE HERE ###

        

    return v

# GRADED FUNCTION: update_parameters_with_momentum



def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):

    """

    Update parameters using Momentum

    

    Arguments:

    parameters -- python dictionary containing your parameters:

                    parameters['W' + str(l)] = Wl

                    parameters['b' + str(l)] = bl

    grads -- python dictionary containing your gradients for each parameters:

                    grads['dW' + str(l)] = dWl

                    grads['db' + str(l)] = dbl

    v -- python dictionary containing the current velocity:

                    v['dW' + str(l)] = ...

                    v['db' + str(l)] = ...

    beta -- the momentum hyperparameter, scalar

    learning_rate -- the learning rate, scalar

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

    v -- python dictionary containing your updated velocities

    """



    L = len(parameters) // 2 # number of layers in the neural networks

    

    # Momentum update for each parameter

    for l in range(L):

        

        ### START CODE HERE ### (approx. 4 lines)

        # compute velocities

        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]

        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        # update parameters

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

        ### END CODE HERE ###

        

    return parameters, v
# GRADED FUNCTION: initialize_adam



def initialize_adam(parameters) :

    """

    Initializes v and s as two python dictionaries with:

                - keys: "dW1", "db1", ..., "dWL", "dbL" 

                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    

    Arguments:

    parameters -- python dictionary containing your parameters.

                    parameters["W" + str(l)] = Wl

                    parameters["b" + str(l)] = bl

    

    Returns: 

    v -- python dictionary that will contain the exponentially weighted average of the gradient.

                    v["dW" + str(l)] = ...

                    v["db" + str(l)] = ...

    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.

                    s["dW" + str(l)] = ...

                    s["db" + str(l)] = ...



    """

    

    L = len(parameters) // 2 # number of layers in the neural networks

    v = {}

    s = {}

    

    # Initialize v, s. Input: "parameters". Outputs: "v, s".

    for l in range(L):

    ### START CODE HERE ### (approx. 4 lines)

        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])

        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])



        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])

        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])

    ### END CODE HERE ###

    

    return v, s
# GRADED FUNCTION: update_parameters_with_adam



def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,

                                beta1=0.9, beta2=0.999, epsilon=1e-8):

    """

    Update parameters using Adam

    

    Arguments:

    parameters -- python dictionary containing your parameters:

                    parameters['W' + str(l)] = Wl

                    parameters['b' + str(l)] = bl

    grads -- python dictionary containing your gradients for each parameters:

                    grads['dW' + str(l)] = dWl

                    grads['db' + str(l)] = dbl

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    learning_rate -- the learning rate, scalar.

    beta1 -- Exponential decay hyperparameter for the first moment estimates 

    beta2 -- Exponential decay hyperparameter for the second moment estimates 

    epsilon -- hyperparameter preventing division by zero in Adam updates



    Returns:

    parameters -- python dictionary containing your updated parameters 

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    """

    

    L = len(parameters) // 2                 # number of layers in the neural networks

    v_corrected = {}                         # Initializing first moment estimate, python dictionary

    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    

    # Perform Adam update on all parameters

    for l in range(L):

        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

        ### START CODE HERE ### (approx. 2 lines)

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]

        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        ### END CODE HERE ###



        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".

        ### START CODE HERE ### (approx. 2 lines)

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))

        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        ### END CODE HERE ###



        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".

        ### START CODE HERE ### (approx. 2 lines)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)

        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        ### END CODE HERE ###



        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".

        ### START CODE HERE ### (approx. 2 lines)

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))

        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        ### END CODE HERE ###



        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".

        ### START CODE HERE ### (approx. 2 lines)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

        ### END CODE HERE ###



    return parameters, v, s
def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,

          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):

    """

    3-layer neural network model which can be run in different optimizer modes.

    

    Arguments:

    X -- input data, of shape (2, number of examples)

    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)

    layers_dims -- python list, containing the size of each layer

    learning_rate -- the learning rate, scalar.

    mini_batch_size -- the size of a mini batch

    beta -- Momentum hyperparameter

    beta1 -- Exponential decay hyperparameter for the past gradients estimates 

    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 

    epsilon -- hyperparameter preventing division by zero in Adam updates

    num_epochs -- number of epochs

    print_cost -- True to print the cost every 1000 epochs



    Returns:

    parameters -- python dictionary containing your updated parameters 

    """



    L = len(layers_dims)             # number of layers in the neural networks

    costs = []                       # to keep track of the cost

    t = 0                            # initializing the counter required for Adam update

    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours

    

    # Initialize parameters

    parameters = initialize_parameters(layers_dims)



    # Initialize the optimizer

    if optimizer == "gd":

        pass # no initialization required for gradient descent

    elif optimizer == "momentum":

        v = initialize_velocity(parameters)

    elif optimizer == "adam":

        v, s = initialize_adam(parameters)

    

    # Optimization loop

    for i in range(num_epochs):

        

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch

        seed = seed + 1

        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)



        for minibatch in minibatches:



            # Select a minibatch

            (minibatch_X, minibatch_Y) = minibatch



            # Forward propagation

            a3, caches = forward_propagation(minibatch_X, parameters)



            # Compute cost

            cost = compute_cost(a3, minibatch_Y)



            # Backward propagation

            grads = backward_propagation(minibatch_X, minibatch_Y, caches)



            # Update parameters

            if optimizer == "gd":

                parameters = update_parameters_with_gd(parameters, grads, learning_rate)

            elif optimizer == "momentum":

                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)

            elif optimizer == "adam":

                t = t + 1 # Adam counter

                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,

                                                               t, learning_rate, beta1, beta2,  epsilon)

        

        # Print the cost every 1000 epoch

        if print_cost and i % 1000 == 0:

            print("Cost after epoch %i: %f" % (i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

                

    # plot the cost

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('epochs (per 100)')

    plt.title("Learning rate = " + str(learning_rate))

    plt.show()



    return parameters
# train 3-layer model

import math

layers_dims = [train_X.shape[0], 5, 2, 1]

parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")



# Predict

predictions = predict(train_X, train_Y, parameters)



# Using momentum optimization method

# train 3-layer model

layers_dims = [train_X.shape[0], 5, 2, 1]

parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")



# Predict

predictions = predict(train_X, train_Y, parameters)



#mini batch using the adam method

# train 3-layer model

layers_dims = [train_X.shape[0], 5, 2, 1]

parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")



# Predict

predictions = predict(train_X, train_Y, parameters)

# as we can see that the adam optimizer gave out an accuracy of about 98 percent which is pretty good right now
