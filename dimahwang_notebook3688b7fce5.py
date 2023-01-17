# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train_data = pd.read_csv("../input/train.csv")

#print(train_data)



# Any results you write to the current directory are saved as output.

def init_parameters(dim_list):

        parameters = {} #parameter is a dictionary that will contain weight and bias parameters

        for layer in range(1, len(dim_list)):

            parameters['W'+ str(layer)] = np.random.randn(dim_list[layer], dim_list[layer -1]) * 0.01

            parameters['b'+ str(layer)] = np.random.randn(dim_list[layer],1)

        return parameters



def propagate(A, W, b):

    Z = np.dot(W, A)+b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    result = (A, W, b) 

    return Z, result



def sigmoid_f(Z):

    A = np.exp(Z)/(1+np.exp(Z))

    cache = (Z)

    return A, cache



def relu_f(Z):

    #print("RELU in: " + str(Z))

    A = np.maximum(0, Z)

    cache = (Z)

    assert(A.shape == Z.shape)

    #print("RELU out: " + str(A))

    return A, cache



def softmax_f(z):

    score = np.exp(z) / np.sum(np.exp(z))

    cache = (z)

    return score,cache



def forward_activation(A_prev, W, b, act_func):

    #print("PROP: "+str(A_prev))

    if act_func == 'sigmoid':

        Z, result = propagate(A_prev, W, b)

        A, cache_act = sigmoid_f(Z)

    elif act_func == 'relu':

        Z, result = propagate(A_prev, W, b)

        A, cache_act = relu_f(Z)

    elif act_func == 'softmax':

        Z, result = propagate(A_prev, W, b)

        A, cache_act = softmax_f(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (result, cache_act)   

    return A, cache



def forward_prop(X, parameters):

    caches = []

    A = X

    L = len(parameters) // 2

    for lay in range(1,L):

        A_prev = A

        A, cache = forward_activation(A_prev, parameters['W'+str(lay)], parameters['b'+str(lay)], 'relu')

        caches.append(cache)    

#    AL, cache = forward_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')

    AL, cache = forward_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], 'softmax')

    caches.append(cache)

    return AL, caches



# A -> activations

# Y ->labels

# Hy′(y):=−∑i(y′ilog(yi)+(1−y′i)log(1−yi)) Cross-entropy loss just for test 

def cross_entropy_loss(A, Y):

    n = Y.shape[0]

    #cost = - np.sum(np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1 -A)))/

    #print("index of 1: "+str(np.where(Y==1)))

    #print("Output of correct labe: "+str(A[1]))

    cost = - np.log(A[1])

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost
#dA/dZ, dL/dZ

def inverse_softmax(dA, cache):

    Z = cache

    s = np.exp(Z) / np.sum(np.exp(Z))

    dZ = dA * s * (1-s)

    assert(dZ.shape==Z.shape)

    return dZ



def inverse_relu(dA, cache):

    Z = cache

    dZ = np.array(Z, copy=True)

    dZ[Z<=0] = 0

    assert(dZ.shape==Z.shape)

    return dZ



def propagate_backward(dZ, cache):

    A_prev, W, b = cache

    dW = np.dot(dZ, A_prev.T)

    db = np.sum(dZ,axis=1,keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)

    assert(dW.shape == W.shape)

    assert(db.shape == b.shape)

    return dW, db, dA_prev



def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == 'softmax':

        dZ = inverse_softmax(dA, activation_cache)

        dW, db, dA = propagate_backward(dZ, linear_cache)   

    elif activation == 'relu':

        dZ = inverse_relu(dA, activation_cache)

        dW, db, dA = propagate_backward(dZ, linear_cache)

    return dW, db, dA



def L_model_backward(AL, Y, caches):

    #for last layer:

    grad = {}

    L = len(caches)

    Y.shape = AL.shape

    dA = AL - Y

    current_cache = caches[L-1]

    grad['dW'+str(L)], grad['db'+str(L)], grad['dA'+str(L)] = linear_activation_backward(dA, current_cache, activation='softmax')

    for layer in reversed(range(L-1)):

        current_cache = caches[layer]

        dW_temp, dB_temp, dA_prev_temp = linear_activation_backward(grad["dA"+str(layer+2)], current_cache, activation='relu')

        grad["dA"+str(layer+1)] = dA_prev_temp

        grad["dW"+str(layer+1)] = dW_temp

        grad["db"+str(layer+1)] = dB_temp

    return grad



def parameter_update(parameter, grad, learn_rate):

    L = len(parameter) // 2

    for layer in range(L):

        parameter['W'+str(layer+1)] = parameter['W'+str(layer+1)] - learn_rate * grad['dW'+str(layer+1)]

        parameter['b'+str(layer+1)] = parameter['b'+str(layer+1)] - learn_rate * grad['db'+str(layer+1)]

    return parameter
def L_NeuralNet(learn_rate = 0.0075, num_iter = 1):

    costs=[]

    #X = np.random.randn(5,1)

    Y = np.array([0,1,0,0,0,0,0,0,0,0])

    parameters = init_parameters([784,100,20,10])

    for i in range(0,num_iter):

        X = np.random.randn(784,1)

        AL, caches = forward_prop(X, parameters)

        cost = cross_entropy_loss(AL,Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = parameter_update(parameters, grads, learn_rate)

        if i%100==0:

            print("Cost: "+str(cost)+"at iteration: "+str(i))



in_data = train_data.as_matrix()

print(in_data.shape)

print(in_data[2][0])

L_NeuralNet()
