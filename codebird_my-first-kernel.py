import numpy as np

import csv

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from random import randint

%matplotlib inline
def linear_forward(A, W, b):

    Z = np.dot(W, A) + b

    

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    

    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":

        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        A, activation_cache = relu(Z)

    elif activation == "tanh":

        A, activation_cache = tanh(Z)

    

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)



    return A, cache
def L_model_forward(X, parameters):

    caches = []

    A = X

    L = len(parameters) // 2

    

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], 'relu')

        caches.append(cache)

    

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], 'sigmoid')

    caches.append(cache)

    

    assert(AL.shape == (10,X.shape[1]))

            

    return AL, caches
def compute_cost(AL, Y, parameters, layers, lambd):

    m = Y.shape[1]

    p = 0

    for i in range(1, layers):

        p += np.sum(np.square(parameters["W" + str(i)]))

    logprobs = np.multiply(-np.log(AL),Y) + (lambd * p)/ (2 * m)

    cost = 1./m * np.nansum(logprobs)

    #cost = (- np.sum(np.dot(Y, np.log(AL.T))) / m) 

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost
def linear_backward(dZ, cache, lambd):

    A_prev, W, b = cache

    m = A_prev.shape[1]



    dW = (np.dot(dZ, A_prev.T) / m) + ((lambd * W) / m) 

    db = np.sum(dZ, axis = 1, keepdims = True) / m

    dA_prev = np.dot(W.T, dZ)

    

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

    elif activation == "tanh":

        dZ = tanh_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches, lambd):

    grads = {}

    L = len(caches) # the number of layers

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 

    

    current_cache = caches

    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache[L-1], activation = "sigmoid", lambd=lambd)

    

    for l in reversed(range(L-1)):

        current_cache = caches

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache[l], activation = "relu", lambd=lambd)

        grads["dA" + str(l + 1)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

    return grads
def L_layer_model(X, Y, layers_dims, data_test,labels_test_dense, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, lambd=0.1):#lr was 0.009

    costs = []                         # keep track of cost

    accuracies = []

    v_accuracies = []

    parameters = initialize_parameters_deep(layers_dims)

    m = Y.shape[1]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y, parameters, len(layers_dims), lambd)

        grads = L_model_backward(AL, Y, caches, lambd)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

            costs.append(cost)

            accuracy = float(np.sum(np.argmax(Y, axis = 0) == np.argmax(AL, axis = 0)))*100.0/float(42000)

            accuracies.append(accuracy)

            print("Train Accuracy: "+str(accuracy))

            validation = predict(parameters, data_test, layers_dims)

            accuracy = (np.sum(validation == labels_test_dense) / testPart) * 100

            v_accuracies.append(accuracy);

            print("Validation Accuracy: " + str(accuracy) + "%")

            

    # plot the cost and accuries

    plt.plot(np.squeeze(costs), label="cost")

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.legend()

    plt.show()

    plt.close()

    plt.plot(np.squeeze(accuracies), label="training accuracy")

    plt.plot(np.squeeze(v_accuracies), label="validation accuracy")

    plt.ylabel('accuracies')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.legend()

    plt.show()

    

    return parameters
def initialize_parameters_deep(layer_dims):

    parameters = {}

    L = len(layer_dims) # number of layers in the network

    

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network



    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads["db" + str(l+1)])

        

    return parameters
def sigmoid(Z):

    A = 1/(1+np.exp(-Z))

    cache = Z

    

    return A, cache
def relu(Z):

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z 

    return A, cache
def sigmoid_backward(dA, cache):

    Z = cache

    

    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    

    assert (dZ.shape == Z.shape)

    

    return dZ
def relu_backward(dA, cache):

    Z = cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    

    # When z <= 0, you should set dz to 0 as well. 

    dZ[Z <= 0] = 0

    

    assert (dZ.shape == Z.shape)

    

    return dZ
def dense_to_1hot(labels, shape1):

    shape0 = labels.shape[0]

    one_hot = np.zeros((shape0, shape1))

    one_hot[np.arange(shape0), labels] = 1

    return one_hot
def predict(parameters, data, layers_dims):

    layers = len(layers_dims)

    A = []

    A.append(data)

    for i in range(1, layers):

        A.append(np.dot(parameters["W" + str(i)], A[i-1]) + parameters["b" + str(i)])

    predictions = np.argmax(A[layers-1], axis=0)

    return predictions
def write_predictions(predictions):

    with open('../input/submission_nn.csv', 'w') as subs:

        subs.write("ImageId,Label\n")

        for i, pred in enumerate(predictions):

            subs.write(str(i+1)+','+str(pred)+'\n')
totalTrain = 42000

tainPart = 40000

testPart = 2000

data = []

with open('../input/train.csv', 'r') as train:

    pixelReader = csv.reader(train, delimiter=',')

    next(pixelReader, None)

    for row in pixelReader:

        data.append(row[0:])

        

data = np.array(data).astype(int)

np.random.shuffle(data)

data_train, data_test = data[:tainPart,:], data[tainPart:,:]

labels_train_dense = data_train[:, 0]

labels_test_dense = data_test[:, 0]



data_train = np.multiply(data_train[:, 1:].T, 1/255)

data_test = np.multiply(data_test[:, 1:].T, 1/255)

assert(data_train.shape == (784, tainPart))

assert(data_test.shape == (784, testPart))



labels_train = dense_to_1hot(labels_train_dense, 10).T

labels_test = dense_to_1hot(labels_test_dense, 10).T

assert(labels_train.shape == (10, tainPart))

assert(labels_test.shape == (10, testPart))



image_to_show = 10

plt.axis('off')

plt.imshow(data_train[:, image_to_show].reshape(28, 28),  cmap=cm.binary)
layers_dims = [784, 40, 30, 10] #  4-layer model

parameters = L_layer_model(data_train, labels_train, layers_dims, data_test, labels_test_dense, num_iterations = 3200, print_cost = True, learning_rate = 0.0075, lambd = 3.3)
test = []

with open('../input/test.csv', 'r') as train:

    pixelReader = csv.reader(train, delimiter=',')

    next(pixelReader, None)

    for row in pixelReader:

        test.append(row[0:])



test = np.array(test).astype(int)

test = np.multiply(test, 1.0 / 255.0)

test = test.T

#uncomment the last line to write your predictions to file

predictions = predict(parameters, test, layers_dims)

#write_predictions(predictions)