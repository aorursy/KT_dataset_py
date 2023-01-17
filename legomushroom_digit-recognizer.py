# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
np.random.seed(1)
def to_one_hot(Y, num_items):

    newY = np.zeros((0, num_items))

    i = 0

    for y in Y:

        newy = np.zeros((1, num_items))

        newy[0, y[0]] = 1

        newY = np.append(newY, newy, axis=0)

    return newY
file_name = '../input/train.csv'

df = pd.read_csv(file_name, header = 0)



original_headers = list(df.columns.values)

numpy_array = df.as_matrix()



numpy_array_random = numpy_array[:, :]

# np.random.shuffle(numpy_array_random)



num_items = 10

Y_orig = numpy_array_random[:, 0:1]

Y = to_one_hot(Y_orig, num_items).T

X = numpy_array_random[:, 1:].T

X_count, m = X.shape

x_count = int(np.sqrt(X_count))



X = ((X - 128) / 128)



trainSize = 25000

devSize = (m - trainSize) // 2

train_X, dev_X, test_X = X[:, :trainSize], X[:, trainSize:(trainSize + devSize)], X[:, (trainSize + devSize):m]

train_Y, dev_Y, test_Y = Y[:, :trainSize], Y[:, trainSize:(trainSize + devSize)], Y[:, (trainSize + devSize):m]



index = 5

item = train_X[:, index]

img_X = item.reshape((x_count, x_count))

plt.imshow(img_X, shape=(x_count, x_count))

print('label:' + str(np.argmax(train_Y[:, index], axis=0)))

print(train_Y[:, index])
def accuracy(p, y):

    return np.mean(p[0, :] == y[0, :])

    end



y = np.matrix([[0, 0, 1, 0, 1, 0, 1]])

p = np.matrix([[1, 0, 1, 0, 1, 1, 1]])



print(metrics.classification_report(y.T, p.T))
def initialize_parameters(shape):

    np.random.seed(3)

    N = shape[0]

    xavier = 1 / N



    parameters = {}

    momentums = {}

    for l in range(1, len(shape)):

        W = (np.random.randn(shape[l], shape[l-1]) * xavier)

        b = np.zeros((shape[l], 1))

        parameters[l] = {

            'W': W,

            'b': b,

        }

        

        momentums[l] = {

            'Vdw': np.zeros(W.shape),

            'Sdw': np.zeros(W.shape),

            'Vdb': np.zeros(b.shape),

            'Sdb': np.zeros(b.shape)

        }

        

        l += 1

    

    return parameters, momentums



# parameters = initialize_parameters(NN_shape)
def relu(Z):

    return np.maximum(0, Z)

    end



def sigmoid(Z):

    return 1 / (1 + np.exp(-Z))

    end
def step_forward(X, W, b, activation_function = 'relu'):

    z = np.dot(W, X) + b

    

    if activation_function == 'relu':

        a = relu(z)

    else:

        a = sigmoid(z)

        

    return a, z
def forward_propagation(X, parameters, NN_shape):

    cache = { 0: { 'A': X } }

    A = X

    L = len(parameters)

    for l in range(1, L+1):

        params = parameters[l]

        if l < L:

            activation_function = 'relu'

        else:

            activation_function = 'sigmoid'

        

#         prev_A = A

        A, Z = step_forward(A, params['W'], params['b'], activation_function)

        cache[l] = { 'A': A, 'Z': Z, 'W': params['W'] }

        

    return A, cache
NN_shape = np.array([X_count, 20, 15, 10])

parameters, _ = initialize_parameters(NN_shape)

A, cache = forward_propagation(X, parameters, NN_shape)

print(A.shape, len(cache))
# def step_backward(Z, A_prev, W, b, dA, activation_function = 'relu'):

def step_backward(dA, A_prev, Z, W, activation_function = 'relu'):

    if activation_function == 'relu':

        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = 0

    elif activation_function == 'sigmoid':

        s = 1 / (1 + np.exp(-Z))

        dZ = dA * s * (1-s)

    # calculate how much should we change previous parameters

    # by taking the previous activations

    # and multiplying them to the current linear activations

    dW = (np.dot(dZ, A_prev.T) / m)

    db = np.sum(dZ, axis=1, keepdims=True) / m

    dA = np.dot(W.T, dZ)

    

    return dA, dW, db
# last_layer_cache = cache[4]

# prev_layer_cache = cache[3]

# last_layer_params = parameters[4]

# dAL = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))



# dA, dW, db = step_backward(last_layer_cache['Z'], prev_layer_cache['A'], last_layer_params['W'], last_layer_params['b'], dAL, 'sigmoid')
def backward_propagation(A, Y, parameters, NN_shape, cache):

    dA = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))

    grads = {}

    for l in range(len(NN_shape)-1, 0, -1):

        c = cache[l]

        prev_c = cache[l-1]

        if l == (len(NN_shape) - 1):

            activation_function = 'sigmoid'

        else:

            activation_function = 'relu'



        params = parameters[l]

        dA, dW, db = step_backward(dA, prev_c['A'], c['Z'], c['W'], activation_function)

        grads[l] = {

            'dW': dW,

            'db': db

        }

        

    return grads
# grads = backward_propagation(A, Y, NN_shape, cache)
import math

import copy



def optimize_parameters(parameters, gradients, learning_rate, momentums, t = 0, beta1 = .9, beta2 = .999):

    eps = 0.00000001

    

    for l in range(0, len(NN_shape) // 2):

        grads = gradients[l]

#         moms = momentums[l]

#         old_params = copy.deepcopy(parameters)

        

        dW = grads['dW']

        db = grads['db']

        

#         Vdw = moms['Vdw']

#         Sdw = moms['Sdw']



#         Vdb = moms['Vdb']

#         Sdb = moms['Sdb']

        

#         Vdw_old = Vdw

#         Vdb_old = Vdb

#         Sdw_old = Sdw

#         Sdb_old = Sdb

                

#         Vdw = (beta1 * Vdw) + ((1 - beta1) * dW)

#         Vdb = (beta1 * Vdb) + ((1 - beta1) * db)

        

#         Sdw = (beta2 * Sdw) + ((1 - beta2) * np.power(dW, 2))

#         Sdb = (beta2 * Sdb) + ((1 - beta2) * np.power(db, 2))

        

#         Vdw_corrected = Vdw / (1 - np.power(beta1, t+1))

#         Vdb_corrected = Vdb / (1 - np.power(beta1, t+1))

        

#         Sdw_corrected = Sdw / (1 - np.power(beta2, t+1))

#         Sdb_corrected = Sdb / (1 - np.power(beta2, t+1))

        

        parameters[l + 1]['W'] = parameters[l + 1]['W'] - (learning_rate * dW)

        parameters[l + 1]['b'] = parameters[l + 1]['b'] - (learning_rate * db)



        

#         parameters[l]['W'] = parameters[l]['W'] - learning_rate * (Vdw_corrected / (np.sqrt(Sdw_corrected) + eps))

#         parameters[l]['b'] = parameters[l]['b'] - learning_rate * (Vdb_corrected / (np.sqrt(Sdb_corrected) + eps))



#         momentums[l]['Vdw'] = Vdw

#         momentums[l]['Vdb'] = Vdb

        

#         momentums[l]['Sdw'] = Sdw

#         momentums[l]['Sdb'] = Sdb

        

    return parameters, momentums
def model(NN_shape, X, Y, learning_rate=0.0001, epoch_size = 1000, num_iterations=2500, epoch_costs_count = 5000):

    # batch gradient descent

    costs = []

    parameters, momentums = initialize_parameters(NN_shape)

    for i in range(0, num_iterations):

        A, cache = forward_propagation(X, parameters, NN_shape)

        gradients = backward_propagation(A, Y, parameters, NN_shape, cache)

        parameters, momentums = optimize_parameters(parameters, gradients, learning_rate, momentums)



#         cost = np.mean(np.sum(((-np.dot(Y,np.log(A).T) - np.dot(1-Y, np.log(1 - A).T)) / m), axis=1, keepdims=True))

        cost = np.mean(np.sum(((Y * np.log(A)) + ((1 - Y) * (np.log(1 - A)))), axis=1, keepdims=True) / -m)

        cost = np.squeeze(cost)  

        

        if ((i % 50) == 0):

            print(str(i) + '>> cost: ' + str(cost))

            costs.append(cost)

            

    return parameters, costs, A 

    

# Adam    

#     costs = []

#     temp_costs = []

#     epoch_i = 0

#     parameters, momentums = initialize_parameters(NN_shape)

#     total = X.shape[1]

#     for i in range(0, num_iterations):

#         epoch_start = 0

#         while epoch_start < total:

#             np.random.seed(epoch_i)

            

#             permutation = np.random.permutation(total)

#             epoch_permutation = np.random.permutation(epoch_size)

#             result_permutation = permutation[epoch_permutation]

            

#             start = epoch_start

#             end = min(epoch_start + epoch_size, total)

#             epoch_start = epoch_start + epoch_size

            

#             x = X[:, result_permutation]

#             y = Y[:, result_permutation]



#             a, cache = forward_propagation(x, parameters, NN_shape)

#             gradients = backward_propagation(a, y, parameters, NN_shape, cache)

#             parameters, momentums = optimize_parameters(parameters, gradients, learning_rate, momentums, epoch_i)

#             cost = np.mean(np.sum(((-np.dot(y,np.log(a).T) - np.dot(1-y, np.log(1 - a).T)) / epoch_size), axis=0, keepdims=True))

#             temp_costs.append(cost)

            

#             learning_rate = learning_rate / 1.0001

#             epoch_i = epoch_i + 1

            

#             if len(temp_costs) == epoch_costs_count:

#                 mean_costs = np.mean(temp_costs)

#                 costs.append(mean_costs)

                

#                 dev_A, dev_cache = forward_propagation(dev_X, parameters, NN_shape)

#                 dev_cost = np.mean(np.sum(((-np.dot(dev_Y,np.log(dev_A).T) - np.dot(1-dev_Y, np.log(1 - dev_A).T)) / dev_A.shape[1]), axis=0, keepdims=True))

                

#                 train_A, train_cache = forward_propagation(X, parameters, NN_shape)

#                 train_cost = np.mean(np.sum(((-np.dot(Y,np.log(train_A).T) - np.dot(1-Y, np.log(1 - train_A).T)) / train_A.shape[1]), axis=0, keepdims=True))

                

#                 print(str(i) + ': cost after ' + str(epoch_costs_count) + '(' + str(epoch_costs_count * epoch_size) + ')' + ' epochs: ' + str(mean_costs) + ' | train cost: ' + str(train_cost) + ' | dev cost: ' + str(dev_cost) + ' || lr: ' + str(learning_rate))

#                 temp_costs = []

            

# #             if ((i % 100) == 0) and (i != 0):

# #                 print(str(i) + '>> cost: ' + str(cost))

            

#     return costs, parameters, a
NN_shape = np.array([X_count, 20, 15, 10])

learning_rate = 0.015

num_iterations = 3500

epoch_size = 25000

epoch_costs_count = 100

params, costs, A = model(NN_shape, train_X, train_Y, learning_rate, epoch_size, num_iterations, epoch_costs_count)
# plot the cost

plt.plot(costs)

plt.ylabel('cost')

plt.xlabel('epochs (per ' + str(epoch_costs_count)  +')')

plt.title("Learning rate = " + str(learning_rate))

plt.show()
# def predict(X, params, NN_shape):

#     A, _ = forward_propagation(X, params, NN_shape)

#     return A
from sklearn import metrics



def predict(X, y, parameters, NN_shape):

    """

    This function is used to predict the results of a  L-layer neural network.

    

    Arguments:

    X -- data set of examples you would like to label

    parameters -- parameters of the trained model

    

    Returns:

    p -- predictions for the given dataset X

    """

    

    m = X.shape[1]

    n = len(parameters) // 2 # number of layers in the neural network

    p = np.zeros((1,m))

    

    # Forward propagation

    probas, caches = forward_propagation(X, parameters, NN_shape)

    

    newY = np.zeros((1, y.shape[1]))

    newProbas = np.zeros((1, probas.shape[1]))

    

    no_equal_cnt = 0

    images_count = 0

    for i in range(0, y.shape[1]):

        newY[0, i] = np.argmax(y[:, i], axis=0)

        newProbas[0, i] = np.argmax(probas[:, i], axis=0)

        

        if newY[0, i] != newProbas[0, i]:

            no_equal_cnt += 1

            

            if images_count < 2:

                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

                item = X[:, i]

                plt.figure()

                plt.imshow(item.reshape((28, 28)), shape=(28, 28))

                plt.show()

                print('prediced label: ' + str(newProbas[0, i]) + ' || real label: ' + str(newY[0, i]))

                images_count += 1

    

    index = 10

    print('predicted: ', newProbas[0, 0:index])

    print('actual:    ', newY[0, 0:index])

#     print(str(newProbas[0, index]) + '/' + str(newY[0, index]))

    print('')

    print('not equal cnt: ' + str(no_equal_cnt))

    print('')

    print('')

    print(metrics.classification_report(newProbas.T, newY.T))

    return probas



pred_dev = predict(dev_X, dev_Y, parameters, NN_shape)