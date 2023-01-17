import pandas as pd

import math

import os



data = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

print(data.shape)

data.head()
y = data.target.values

x_data = data.drop(['target','ID_code'], axis=1)
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(x_data)  

x_data = scaler.transform(x_data)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.1, random_state=42)
y_train= y_train.reshape((y_train.shape[0], 1))

y_test= y_test.reshape((y_test.shape[0] , 1))



x_train=x_train.T

y_train=y_train.T



print(x_train.shape)

print(y_train.shape)
import numpy as np

import matplotlib.pyplot as plt
def initialize_parameters(layer_dims):

    parameters = {}

    L = len(layer_dims)           

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01

        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

    return parameters  
def linear_forward(A, W, b):

    Z = np.dot(W,A)+b

    cache = (A, W, b)

    return Z, cache
def sigmoid(z):

    A = 1/(1 + np.exp(-z))

    activation_cache = A.copy()

    return A, activation_cache
def relu(z):

    A = z*(z > 0)

    activation_cache = z

    return A, activation_cache
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
def forward_propagation(minibatch_X, parameters):

    caches = []

    A = minibatch_X

    L = len(parameters) // 2                  

    for l in range(1, L):

        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")

        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches
def compute_cost(AL, Y,parameters,lambd ):

    m = Y.shape[1]

    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))

    cost = np.squeeze(cost)  

    

    L = len(parameters) // 2 

    regularization = 0;

    

    for l in range(L):

        regularization +=  np.sum(np.square(parameters["W" + str(l + 1)]))

        

    L2_regularization_cost = lambd / (2 * m) * regularization

    cost = cost + L2_regularization_cost

    return cost
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    np.random.seed(seed)  

    m = X.shape[1] 

    mini_batches = []



    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((1, m))

    

    num_complete_minibatches = math.floor(m / mini_batch_size) 

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)



    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]

        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)



    return mini_batches
def initialize_velocity(parameters):

    L = len(parameters) // 2  

    v = {}

    for l in range(L):

        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)

        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):

    L = len(parameters) // 2  

    for l in range(L):

        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]

        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v
def initialize_adam(parameters):

    L = len(parameters) // 2 

    v = {}

    s = {}

    for l in range(L):

        v["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)

        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

        s["dW" + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)

        s["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v, s
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,beta1=0.9, beta2=0.999, epsilon=1e-8):

    L = len(parameters) // 2  

    v_corrected = {}  

    s_corrected = {}  



    for l in range(L):

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]

        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]



        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))

        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))



        s["dW" + str(l + 1)] = beta1 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)

        s["db" + str(l + 1)] = beta1 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)



        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))

        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))



        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (

        v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon))

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (

        v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon))

    return parameters, v, s
def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]

        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters
def linear_backward(dZ, cache,lambd):

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,A_prev.T) + lambd / m * W

    db = (1/m)*np.sum(dZ, axis=1, keepdims = True)

    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db
def sigmoid_backward(dA, activation_cache):

    return dA*(activation_cache*(1-activation_cache))
def relu_backward(dA, activation_cache):

    return dA*(activation_cache > 0)
    

def linear_activation_backward(dA, cache, activation, lambd):

    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)    

    return dA_prev, dW, db
def backward_propagation(minibatch_X, minibatch_Y, caches, lambd):

    grads = {}

    L = len(caches) 

    m = minibatch_X.shape[1]

    minibatch_Y = minibatch_Y.reshape(1,minibatch_X.shape[1])

    dAL = - (np.divide(minibatch_Y, minibatch_X) - np.divide(1 - minibatch_Y, 1 - minibatch_X))

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)

    for l in reversed(range(L-1)):

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu", lambd)

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

    return grads
def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,beta1=0.9, beta2=0.999, epsilon=1e-8,lambd = 0.7,  num_epochs=10000, print_cost=True):

    L = len(layers_dims)  

    costs = []  

    t = 0  

    seed = 10  

    parameters = initialize_parameters(layers_dims)



    if optimizer == "gd":

        pass  

    elif optimizer == "momentum":

        v = initialize_velocity(parameters)

    elif optimizer == "adam":

        v, s = initialize_adam(parameters)



    for i in range(num_epochs):

        seed = seed + 1

        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            AL, caches = forward_propagation(minibatch_X, parameters)

            cost = compute_cost(AL, minibatch_Y,parameters,lambd )

            grads = backward_propagation(AL, minibatch_Y, caches, lambd)

            if optimizer == "gd":

                parameters = update_parameters_with_gd(parameters, grads, learning_rate)

            elif optimizer == "momentum":

                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)

            elif optimizer == "adam":

                t = t + 1  

                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,t, learning_rate, beta1, beta2, epsilon)

        if print_cost and i % 1 == 0:

            print("Cost after epoch %i: %f" % (i, cost))

        if print_cost and i % 1 == 0:

            costs.append(cost)



    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('epochs (per 100)')

    plt.title("Learning rate = " + str(learning_rate))

    plt.show()



    return parameters
def predict(X, y, parameters): 

    m = X.shape[1]

    p = np.zeros((1,m), dtype = np.int)

    AL, caches = forward_propagation(X, parameters)

    for i in range(0, AL.shape[1]):

        if AL[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    print(str(np.mean((p[0,:] == y[0,:]))))

    return p
layers_dims =np.array([200,1000,1000,1000,1000,1000,1])

parameters = model(x_train, y_train, layers_dims, optimizer="adam", learning_rate=0.001, mini_batch_size=10000, beta=0.9,beta1=0.9, beta2=0.999, epsilon=1e-8,lambd = 13.24, num_epochs=100, print_cost=True)

pred_train = predict(x_train, y_train, parameters)

print(pred_train)

pred_test = predict(x_test.T, y_test.T, parameters)

print(pred_test)