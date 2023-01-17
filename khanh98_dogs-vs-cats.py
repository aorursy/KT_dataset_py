import os
def list_files(path):
    files = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(dirname, filename)
            files.append(name)
    return files
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

list_files('/kaggle/input')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile
def unzip(zip_path, unzip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)
pwd
dataset_name = 'dogs-vs-cats'
test_zip_path = f'/kaggle/input/{dataset_name}/test1.zip'
train_zip_path = f'/kaggle/input/{dataset_name}/train.zip'
test_unzip_path = '.' #unzip to working space
train_unzip_path = '.' #unzip to working space
unzip(test_zip_path, test_unzip_path)
unzip(train_zip_path, train_unzip_path)
list_files('.')
label = pd.read_csv(f'/kaggle/input/{dataset_name}/sampleSubmission.csv')
label.head(20)
id_label = label.to_numpy()
id_label.shape
image_size = (64, 64)
def read_data(path):
    data = np.array([])
    Y = np.array([])
    from PIL import Image
    from numpy import asarray
    file_names = list_files(path)
    for file_name in file_names[:100]:
        img = Image.open(file_name)
        img = img.resize(image_size) #shape (64, 64, 3)
        img = asarray(img)
        img = np.expand_dims(img, axis = 0) #shape (1, 64, 64, 3)

        if not data.any():
            data = img
        else:
            data = np.append(data, img, axis=0)
        l = 1 # default is dog
        if file_name.find('cat') > 0:
            l = 0
        print(file_name, ' ', l)
        Y = np.append(Y, [l])

    X = data.reshape(data.shape[0], -1).T    
    return data, X, Y
data_train, X_train, Y_train = read_data('./train')
data_test, X_test, Y_test = read_data('./test1')
print(data_train.shape)
print(X_train.shape)
print(Y_train.shape)
print(Y_train)
import matplotlib.pyplot as plt
i =36



plt.imshow(X_train[:,i].reshape(64,64,3))
print(Y_train[i])
def sigmoid(Z):
    """
    Z can be a scalar, vector or matrix
    """
    result = 1/(1 + np.exp(-Z))
    return result, Z
def sigmoid_backward(Z):
    A, _ = sigmoid(Z)
    return A*(1-A)
def relu(Z):
    new_Z = np.maximum(Z, 0)
    return new_Z, Z
def relu_backward(Z):
    A, _ = relu(Z)
    A[A > 0] = 1
    return A
def initialize_parameters(layer_dims):
    """
    layer_dims contains the number of unit for each layer in network, start with input layer
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
import numpy as np
initialize_parameters([3,4,5])
def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache
def linear_forward_activation(A_prev, W, b, activation):
    """
    A_prev is vector A of last layer
    
    activation has two types: relu or sigmoid
    """
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b) # linear_cache contains A,W, b
        A, activation_cache = relu(Z) # activation_cache contains Z
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)    
    return A, cache
def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2 # number of layer. because each layer except input layer has two params, W and b
    caches = [] # list contains cache for each layer
    # from layer 1 to layer L-1 (relu layer)
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_forward_activation(A_prev, W, b, 'relu')
        caches.append(cache)
    # for last layer (sigmoid layers)
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_forward_activation(A, W, b, 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches
def test_forward_model():
    X = np.array([[1,2,3,4],[4,5,6,7],[7,8,9, 10]])
    Y = np.array([[1,0,1,0]])
    parameters = initialize_parameters([3,4,1])
    print(parameters)
    AL, caches = L_model_forward(X, parameters)
    print(AL)
test_forward_model()
def compute_cost(AL, Y):
    """
    AL is column vector of A in last layer
    """
    m = Y.shape[0] # number of data point
    cost = -1/m*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = cost.squeeze()
    assert(cost.shape == ())
    return cost
print( Y_train.shape[0])
print(compute_cost(np.array([[0.50008811,0.50014391,0.50019507,0.50024622]]), np.array([1,0,1,1])))
def linear_backward(dZ, cache):
    """
    with a specific layer,
    dZ is derrivative of loss function (J) respect to Z of this layer
    cache is linear_cache which contains A_prev, W, b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m*dZ.dot(A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T.dot(dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
def test_linear_backward():
    X = np.array([[1,2,3,4],[4,5,6,7],[7,8,9, 10]])
    Y = np.array([[1,0,1,0]])
    parameters = initialize_parameters([3,4,1])
    AL, caches = L_model_forward(X, parameters)
    
    linear, activation = caches[1]
    A, W, b = linear
    Z = activation
    dA_prev, dW, db = linear_backward(Z, linear)
    assert(dA_prev.shape == A.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
test_linear_backward()
def linear_backward_activation(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    A, W, b = linear_cache
    if activation == 'relu':
        dZ = dA*relu_backward(Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == 'sigmoid':
        dZ = dA*sigmoid_backward(Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #print(dA.shape)
    #print(dZ.shape)
    #print(Z.shape)
    
    
    return dA_prev, dW, db
def test_linear_backward_activation():
    X = np.array([[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    Y = np.array([[1,0,1,0,1,0,0,0,1,1]])
    parameters = initialize_parameters([2,5,3,1])
    AL, caches = L_model_forward(X, parameters)
    
    linear, activation = caches[0]
    A_prev, W, b = linear
    Z = activation
    dA_prev, dW, db = linear_backward_activation(sigmoid(Z)[0], caches[0], 'sigmoid')
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(True == False)
test_linear_backward_activation()
def L_model_backward(AL, Y, caches):
    
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    grads = {} # contains dW, db for each layer
    #derrivative of loss function respects to A of last layer
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #last layer - sigmoid layer
    dA_prev, dW, db = linear_backward_activation(dAL, caches[L-1], 'sigmoid')
    grads['dA' + str(L-1)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db
    
    #from L-1 layer back to layer 1
    for l in reversed(range(L-1)):
        dA_prev, dW, db = linear_backward_activation(grads['dA' + str(l+1)], caches[l], 'relu')
        grads['dA' + str(l)] = dA_prev
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db
    return grads
def update_paramenters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters['W' + str(l)] -= learning_rate*grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate*grads['db' + str(l)]
    return parameters
# 64x64x3=12288


def L_layer_model(X, Y, layer_dims, num_iterations=100, learing_rate=0.01):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(epochs):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_paramenters(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters
feature_nums = image_size[0] * image_size[1] * 3
layer_dims = [feature_nums,20, 7, 5, 1] #  4-layer model]
learning_rate = 0.0075
epochs = 3000
X_train = X_train / 255
X_test = X_test / 255
parameters = L_layer_model(X_train[:,:], Y_train[:], layer_dims, epochs, learning_rate)
def test(X, Y, parameters):
    L = len(parameters) // 2
    AL, caches = L_model_forward(X, parameters)
    AL[AL > 0.5] = 1
    AL[AL <= 0.5] = 0
    print(AL)
    print(Y)
    accurary = np.sum(AL == Y)/len(Y)
    return accurary
print(test(X_train, Y_train, parameters))
