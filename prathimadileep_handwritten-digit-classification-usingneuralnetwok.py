import numpy as np

import pickle

import gzip

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import h5py

import sklearn

import sklearn.datasets

import scipy

from PIL import Image

from scipy import ndimage





%matplotlib inline
def load_data():

    f = gzip.open('../input/dsmnist/mnist.pkl', 'rb')

    f.seek(0)

    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    f.close()

    return (training_data, validation_data, test_data)
training_data, validation_data, test_data = load_data()
training_data
# shape of data

print(training_data[0].shape)

print(training_data[1].shape)
print("The feature dataset is:" + str(training_data[0]))

print("The target dataset is:" + str(training_data[1]))

print("The number of examples in the training dataset is:" + str(len(training_data[0])))

print("The number of points in a single input is:" + str(len(training_data[0][1])))
def one_hot(j):

    # input is the target dataset of shape (m,) where m is the number of data points

    # returns a 2 dimensional array of shape (10, m) where each target value is converted to a one hot encoding

    # Look at the next block of code for a better understanding of one hot encoding

    n = j.shape[0]

    new_array = np.zeros((10, n))

    index = 0

    for res in j:

        new_array[res][index] = 1.0

        index = index + 1

    return new_array
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(data.shape)

one_hot(data)
def data_wrapper():

    tr_d, va_d, te_d = load_data()

    

    training_inputs = np.array(tr_d[0][:]).T

    training_results = np.array(tr_d[1][:])

    train_set_y = one_hot(training_results)

    

    validation_inputs = np.array(va_d[0][:]).T

    validation_results = np.array(va_d[1][:])

    validation_set_y = one_hot(validation_results)

    

    test_inputs = np.array(te_d[0][:]).T

    test_results = np.array(te_d[1][:])

    test_set_y = one_hot(test_results)

    

    return (training_inputs, train_set_y, test_inputs, test_set_y)
train_set_x, train_set_y, test_set_x, test_set_y = data_wrapper()
print ("train_set_x shape: " + str(train_set_x.shape))

print ("train_set_y shape: " + str(train_set_y.shape))

print ("test_set_x shape: " + str(test_set_x.shape))

print ("test_set_y shape: " + str(test_set_y.shape))
y = pd.DataFrame(train_set_y)
print("The target dataset is:" + str(training_data[1]))

print("The one hot encoding dataset is:")

y
index  = 4999

k = train_set_x[:,index]

k = k.reshape((28, 28))

plt.title('Label is {label}'.format(label= training_data[1][index]))

plt.imshow(k, cmap='gray')
def sigmoid(Z):

    

    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 

    # sigmoid_memory is stored as it is used later on in backpropagation

    

    H = 1/(1+np.exp(-Z))

    sigmoid_memory = Z

    

    return H, sigmoid_memory
Z = np.arange(8).reshape(4,2)

print ("sigmoid(Z) = " + str(sigmoid(Z)))
def relu(Z):

    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 

    # relu_memory is stored as it is used later on in backpropagation

    

    H = np.maximum(0,Z)

    

    assert(H.shape == Z.shape)

    

    relu_memory = Z 

    return H, relu_memory
Z = np.array([1, 3, -1, -4, -5, 7, 9, 18]).reshape(4,2)

print ("relu(Z) = " + str(relu(Z)))
def softmax(Z):

    # Z is numpy array of shape (n, m) where n is number of neurons in the layer and m is the number of samples 

    # softmax_memory is stored as it is used later on in backpropagation

   

    Z_exp = np.exp(Z)



    Z_sum = np.sum(Z_exp,axis = 0, keepdims = True)

    

    H = Z_exp/Z_sum  #normalising step

    softmax_memory = Z

    

    return H, softmax_memory
Z = np.array([[11,19,10], [12, 21, 23]])
#Z = np.array(np.arange(30)).reshape(10,3)

H, softmax_memory = softmax(Z)

print(H)

print(softmax_memory)
def initialize_parameters(dimensions):



    # dimensions is a list containing the number of neuron in each layer in the network

    # It returns parameters which is a python dictionary containing the parameters "W1", "b1", ..., "WL", "bL":



    np.random.seed(2)

    parameters = {}

    L = len(dimensions)            # number of layers in the network + 1



    for l in range(1, L): 

        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * 0.1

        parameters['b' + str(l)] = np.zeros((dimensions[l], 1)) 

        

        assert(parameters['W' + str(l)].shape == (dimensions[l], dimensions[l-1]))

        assert(parameters['b' + str(l)].shape == (dimensions[l], 1))



        

    return parameters
dimensions  = [784, 3,7,10]

parameters = initialize_parameters(dimensions)

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))

# print("W3 = " + str(parameters["W3"]))

# print("b3 = " + str(parameters["b3"]))
#Graded



def layer_forward(H_prev, W, b, activation = 'relu'):



    # H_prev is of shape (size of previous layer, number of examples)

    # W is weights matrix of shape (size of current layer, size of previous layer)

    # b is bias vector of shape (size of the current layer, 1)

    # activation is the activation to be used for forward propagation : "softmax", "relu", "sigmoid"



    # H is the output of the activation function 

    # memory is a python dictionary containing "linear_memory" and "activation_memory"

    

    if activation == "sigmoid":

        Z = np.dot(W,H_prev) + b

        linear_memory = (H_prev, W, b)

        H, activation_memory = sigmoid(Z)

 

    elif activation == "softmax":

        Z = np.dot(W,H_prev) + b

        linear_memory = (H_prev, W, b)

        H, activation_memory = softmax(Z)

    

    elif activation == "relu":

        Z = np.dot(W,H_prev) + b

        linear_memory = (H_prev, W, b)

        H, activation_memory = relu(Z)

        

    assert (H.shape == (W.shape[0], H_prev.shape[1]))

    memory = (linear_memory, activation_memory)



    return H, memory
# verify

# l-1 has two neurons, l has three, m = 5

# H_prev is (l-1, m)

# W is (l, l-1)

# b is (l, 1)

# H should be (l, m)

H_prev = np.array([[1,0, 5, 10, 2], [2, 5, 3, 10, 2]])

W_sample = np.array([[10, 5], [2, 0], [1, 0]])

b_sample = np.array([10, 5, 0]).reshape((3, 1))



H = layer_forward(H_prev, W_sample, b_sample, activation="sigmoid")[0]

H
#Graded



def L_layer_forward(X, parameters):



    # X is input data of shape (input size, number of examples)

    # parameters is output of initialize_parameters()

    

    # HL is the last layer's post-activation value

    # memories is the list of memory containing (for a relu activation, for example):

    # - every memory of relu forward (there are L-1 of them, indexed from 1 to L-1), 

    # - the memory of softmax forward (there is one, indexed L) 



    memories = []

    H = X

    L = len(parameters) // 2                  # number of layers in the neural network

    

    # Implement relu layer (L-1) times as the Lth layer is the softmax layer

    for l in range(1, L):

        H_prev = H 

        W = 'W' + str(l)

        b = 'b' + str(l)

        H, memory = layer_forward(H_prev, parameters[W], parameters[b], activation = 'relu')

        

        memories.append(memory)

    

    # Implement the final softmax layer

    # HL here is the final prediction P as specified in the lectures

    W = 'W' + str(L)

    b = 'b' + str(L)

    HL, memory = layer_forward(H, parameters[W], parameters[b], activation = 'softmax')

    

    memories.append(memory)



    assert(HL.shape == (10, X.shape[1]))

            

    return HL, memories
# verify

# X is (784, 10)

# parameters is a dict

# HL should be (10, 10)

x_sample = train_set_x[:, 10:20]

print(x_sample.shape)

HL = L_layer_forward(x_sample, parameters=parameters)[0]

print(HL[:, :5])
#Graded



def compute_loss(HL, Y):





    # HL is probability matrix of shape (10, number of examples)

    # Y is true "label" vector shape (10, number of examples)



    # loss is the cross-entropy loss



    m = Y.shape[1]



    loss = (-1.0/m)*(np.sum(np.multiply(np.log(HL),Y)))

        

    loss = np.squeeze(loss)      # To make sure that the loss's shape is what we expect (e.g. this turns [[17]] into 17).

    assert(loss.shape == ())

    

    return loss
# sample

# HL is (10, 5), Y is (10, 5)

np.random.seed(2)

HL_sample = np.random.rand(10,5)

Y_sample = train_set_y[:, 10:15]

print(HL_sample)

print(Y_sample)



print(compute_loss(HL_sample, Y_sample))
def sigmoid_backward(dH, sigmoid_memory):

    

    # Implement the backpropagation of a sigmoid function

    # dH is gradient of the sigmoid activated activation of shape same as H or Z in the same layer    

    # sigmoid_memory is the memory stored in the sigmoid(Z) calculation

    

    Z = sigmoid_memory

    

    H = 1/(1+np.exp(-Z))

    dZ = dH * H * (1-H)

    

    assert (dZ.shape == Z.shape)

    

    return dZ
def relu_backward(dH, relu_memory):

    

    # Implement the backpropagation of a relu function

    # dH is gradient of the relu activated activation of shape same as H or Z in the same layer    

    # relu_memory is the memory stored in the sigmoid(Z) calculation

    

    Z = relu_memory

    dZ = np.array(dH, copy=True) # dZ will be the same as dA wherever the elements of A weren't 0

    

    dZ[Z <= 0] = 0

    

    assert (dZ.shape == Z.shape)

    

    return dZ
#Graded



def layer_backward(dH, memory, activation = 'relu'):

    

    # takes dH and the memory calculated in layer_forward and activation as input to calculate the dH_prev, dW, db

    # performs the backprop depending upon the activation function

    



    linear_memory, activation_memory = memory

    

    if activation == "relu":

        dZ = relu_backward(dH, activation_memory)

        H_prev, W, b = linear_memory

        m = H_prev.shape[1]

        dW = (1./m)*(np.dot(dZ,H_prev.T))

        db = (1./m)*(np.sum(dZ,axis=1,keepdims=True))

        dH_prev = np.dot(W.T,dZ)

        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dH, activation_memory)

        H_prev, W, b = linear_memory

        m = H_prev.shape[1]

        dW = (1./m)*(np.dot(dZ,H_prev.T))

        db = (1./m)*(np.sum(dZ,axis=1,keepdims=True))

        dH_prev = np.dot(W.T,dZ)

            

    return dH_prev, dW, db
# verify

# l-1 has two neurons, l has three, m = 5

# H_prev is (l-1, m)

# W is (l, l-1)

# b is (l, 1)

# H should be (l, m)

H_prev = np.array([[1,0, 5, 10, 2], [2, 5, 3, 10, 2]])

W_sample = np.array([[10, 5], [2, 0], [1, 0]])

b_sample = np.array([10, 5, 0]).reshape((3, 1))



H, memory = layer_forward(H_prev, W_sample, b_sample, activation="relu")

np.random.seed(2)

dH = np.random.rand(3,5)

dH_prev, dW, db = layer_backward(dH, memory, activation = 'relu')

print('dH_prev is \n' , dH_prev)

print('dW is \n' ,dW)

print('db is \n', db)
#Graded



def L_layer_backward(HL, Y, memories):

    

    # Takes the predicted value HL and the true target value Y and the 

    # memories calculated by L_layer_forward as input

    

    # returns the gradients calulated for all the layers as a dict



    gradients = {}

    L = len(memories) # the number of layers

    m = HL.shape[1]

    Y = Y.reshape(HL.shape) # after this line, Y is the same shape as AL

    

    # Perform the backprop for the last layer that is the softmax layer

    current_memory = memories[-1]

    linear_memory, activation_memory = current_memory

    dZ = HL - Y

    H_prev, W, b = linear_memory

    # Use the expressions you have used in 'layer_backward'

    gradients["dH" + str(L-1)] = np.dot(W.T,dZ)

    gradients["dW" + str(L)] = (1./m)*(np.dot(dZ,H_prev.T))

    gradients["db" + str(L)] = (1./m)*(np.sum(dZ,axis=1,keepdims=True))

    

    # Perform the backpropagation l-1 times

    for l in reversed(range(L-1)):

        # Lth layer gradients: "gradients["dH" + str(l + 1)] ", gradients["dW" + str(l + 2)] , gradients["db" + str(l + 2)]

        current_memory = memories[l]

        dH = 'dH' + str(l+1)

        dH_prev_temp, dW_temp, db_temp = layer_backward(gradients[dH], current_memory, activation = 'relu')

        gradients["dH" + str(l)] = dH_prev_temp

        gradients["dW" + str(l + 1)] = dW_temp

        gradients["db" + str(l + 1)] = db_temp 





    return gradients
# verify

# X is (784, 10)

# parameters is a dict

# HL should be (10, 10)

x_sample = train_set_x[:, 10:20]

y_sample = train_set_y[:, 10:20]



HL, memories = L_layer_forward(x_sample, parameters=parameters)

gradients  = L_layer_backward(HL, y_sample, memories)

print('dW3 is \n', gradients['dW3'])

print('db3 is \n', gradients['db3'])

print('dW2 is \n', gradients['dW2'])

print('db2 is \n', gradients['db2'])
#Graded



def update_parameters(parameters, gradients, learning_rate):



    # parameters is the python dictionary containing the parameters W and b for all the layers

    # gradients is the python dictionary containing your gradients, output of L_model_backward

    

    # returns updated weights after applying the gradient descent update



    

    L = len(parameters) // 2 # number of layers in the neural network



    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients["dW" + str(l+1)]

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradients["db" + str(l+1)]



        

    return parameters
dimensions = [784, 45, 10] #  three-layer model
#Graded



def L_layer_model(X, Y, dimensions, learning_rate = 0.0075, num_iterations = 3000, print_loss=False):

    

    # X and Y are the input training datasets

    # learning_rate, num_iterations are gradient descent optimization parameters

    # returns updated parameters



    np.random.seed(2)

    losses = []                         # keep track of loss

    

    # Parameters initialization

    parameters = initialize_parameters(dimensions)

 

    for i in range(0, num_iterations):



        # Forward propagation

        HL, memories = L_layer_forward(X, parameters)

        

        # Compute loss

        loss = compute_loss(HL, Y)

    

        # Backward propagation

        gradients = L_layer_backward(HL, Y, memories)

 

        # Update parameters.

        parameters = update_parameters(parameters, gradients, learning_rate)

                

        # Printing the loss every 100 training example

        if print_loss and i % 100 == 0:

            print ("Loss after iteration %i: %f" %(i, loss))

            losses.append(loss)

            

    # plotting the loss

    plt.plot(np.squeeze(losses))

    plt.ylabel('loss')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
train_set_x_new = train_set_x[:,0:5000]

train_set_y_new = train_set_y[:,0:5000]

train_set_x_new.shape
parameters = L_layer_model(train_set_x_new, train_set_y_new, dimensions, num_iterations = 2000, print_loss = True)
def predict(X, y, parameters):

    

    # Performs forward propogation using the trained parameters and calculates the accuracy

    

    m = X.shape[1]

    n = len(parameters) // 2 # number of layers in the neural network

    

    # Forward propagation

    probas, caches = L_layer_forward(X, parameters)

    

    p = np.argmax(probas, axis = 0)

    act = np.argmax(y, axis = 0)



    print("Accuracy: "  + str(np.sum((p == act)/m)))

        

    return p
pred_train = predict(train_set_x_new, train_set_y_new, parameters)
pred_test = predict(test_set_x, test_set_y, parameters)
index  = 3474

k = test_set_x[:,index]

k = k.reshape((28, 28))

plt.title('Label is {label}'.format(label=(pred_test[index], np.argmax(test_set_y, axis = 0)[index])))

plt.imshow(k, cmap='gray')