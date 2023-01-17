# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
orig_dataset = pd.read_csv('../input/pulsar_stars.csv')
orig_dataset.head()
orig_dataset.info()
orig_dataset.describe()
orig_dataset['target_class'].value_counts().plot(kind = 'bar', title = 'Count (target_class)')
plt.show()
X = np.array(orig_dataset.iloc[:, 0:-1])
Y = np.array(orig_dataset.iloc[:, -1])
from imblearn.over_sampling import RandomOverSampler

X = X.copy()
Y = Y.copy()

ros = RandomOverSampler()
X, Y = ros.fit_sample(X, Y)

Y = Y.reshape(-1,1)

print(X.shape)
print(Y.shape)
df_Y = pd.DataFrame(Y)
df_Y.columns = ['target_class']

df_Y.target_class.value_counts().plot(kind = 'bar', title = 'Count (target_class)')
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

stratified_split = StratifiedShuffleSplit(n_splits = 2, random_state = 0, test_size = 0.2)

for train_index, test_index in stratified_split.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
from sklearn.preprocessing import StandardScaler

std_sclr = StandardScaler()
X_train = std_sclr.fit_transform(X_train)
X_test = std_sclr.fit_transform(X_test)
X_train = X_train.T
Y_train = Y_train.reshape(-1, 1).T

X_test = X_test.T
Y_test = Y_test.reshape(-1, 1).T

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

def initialize_parameters(layer_dims):
    
    parameters = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters
def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
def sigmoid(Z):
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z 
    assert(A.shape == Z.shape)
    
    return A, cache
def relu(Z):
    
    A = np.maximum(0, Z)
    cache = Z
    assert(A.shape == Z.shape)
    
    return A, cache
def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == 'sigmoid':
        Z, lin_cache = linear_forward(A_prev, W, b)
        A, act_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, lin_cache = linear_forward(A_prev, W, b)
        A, act_cache = relu(Z)
        
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (lin_cache, act_cache)
    
    return A, cache
def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches
def compute_cost(AL, Y):
    
    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost
def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dW, db
def sigmoid_backward(dA, cache):
    
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)
    
    return dZ
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    
    return dZ
def linear_activation_backward(dA, cache, activation):
    
    lin_cache, act_cache = cache
    
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, act_cache)
        dA_prev, dW, db = linear_backward(dZ, lin_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, act_cache)
        dA_prev, dW, db = linear_backward(dZ, lin_cache)
        
    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    curr_cache = linear_activation_backward(AL, caches[L - 1], activation = 'sigmoid')
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = curr_cache
    
    for l in reversed(range(L - 1)):
        curr_cache = linear_activation_backward(grads['dA' + str(l + 2)], caches[l], activation = 'relu')
        grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = curr_cache
        
    return grads
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
        
    return parameters
def L_layer_model(X, Y, layer_dims, learning_rate, iterations, print_cost = False):
    
    costs = []
    parameters = initialize_parameters(layer_dims)
    
    for i in range(0, iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            costs.append(cost)
            print('Cost after iteration %i: %f' %(i, cost))
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title('Learning Rate: ' + str(learning_rate))
    plt.show()
    
    return parameters
layer_dims = [X_train.shape[0], 64, 32, 1]
learning_rate = 0.0005
num_iterations = 2500

parameters = L_layer_model(X_train, Y_train, layer_dims = layer_dims, learning_rate = learning_rate, iterations = num_iterations, print_cost = True)
def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))
    
    probs, caches = L_model_forward(X, parameters)

    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
            
    print('Accuracy: ', str(np.sum((p == y)/ m)))
    
    return p
predictions = predict(X_test, Y_test, parameters)
preds = pd.DataFrame(predictions.T)
trues = pd.DataFrame(Y_test.T)

preds.columns = ['predictions']
sum(preds.predictions == 0)
trues.columns = ['predictions']
sum(trues.predictions == 1)
import keras

new_x_train = X_train.copy().T
new_y_train = Y_train.copy().T
new_x_test = X_test.copy().T
new_y_test = Y_test.copy().T
print(new_x_train.shape)
print(new_y_train.shape)
print(new_x_test.shape)
print(new_y_test.shape)
def build_keras_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation = 'relu', input_shape = (new_x_train.shape[1],)))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
    
    return model
model = build_keras_model()
model.summary()
callbacks = [keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3, restore_best_weights = True)]
history = model.fit(new_x_train, new_y_train, epochs = 100, callbacks = callbacks)
new_preds = model.predict(new_x_test)
from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(new_y_test.argmax(axis = 1), new_preds.argmax(axis = 1))
conf_mat = confusion_matrix(new_y_test.argmax(axis = 1), new_preds.argmax(axis = 1))

print('Accuracy: {}%'.format(acc * 100))
print(conf_mat)
preds_df = pd.DataFrame(new_preds.argmax(axis = 1))
trues_df = pd.DataFrame(new_y_test)

preds_df.to_csv('preds.csv', index = False)
trues_df.to_csv('trues.csv', index = False)
