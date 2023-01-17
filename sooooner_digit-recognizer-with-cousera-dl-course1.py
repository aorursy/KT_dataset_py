import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')
df_train.tail()
df_test.tail()
dfY = df_train["label"]
dfY = pd.get_dummies(dfY)
dfY = dfY.as_matrix().reshape(len(dfY), -1).T
dfX = df_train.drop("label", axis = 1)
dfX /= np.max(np.max(dfX))
dfX = dfX.as_matrix().reshape(len(dfX), -1).T
df_test /= np.max(np.max(df_test))
df_test = df_test.as_matrix().reshape(len(df_test), -1).T
del df_train
np.shape(dfX), np.shape(dfY), np.shape(df_test)
plt.figure(figsize=(6, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(dfX[:, i].reshape(28,28))
    plt.axis('off')
    plt.title(np.where(dfY[:, i] == 1)[0][0])
plt.show()
plt.figure(figsize=(6, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(df_test[:, i].reshape(28,28))
    plt.axis('off')
plt.show()
sns.barplot(list(range(10)), np.sum(dfY, axis=1), palette='Greens_d')
print(np.sum(dfY, axis=1))
#check None Value
# assert(dfX.isnull().any().count() == np.shape(dfX)[1])
# assert(dfY.isnull().any().count() == np.shape(dfY)[1])
def layer_sizes(dfX, dfY):
    """
    Arguments:
    dfX -- input dataset of shape (input size, number of examples)
    dfY -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_hi -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = dfX.shape[0]      # size of input layer
    n_h1 = 128
    n_h2 = 128
    n_h3 = 128
    n_y = dfY.shape[0]      # size of output layer
    layer_dims = (n_x, n_h1, n_h2, n_h3, n_y)
    
    return layer_dims
layer_dims = layer_sizes(dfX, dfY)
layer_dims
def initialize_parameters(layer_dims):
    """
    Argument:
    layer_dims -- output of layer_sizes()
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h1, n_x)
                    b1 -- bias vector of shape (n_h1, 1)
                    W2 -- weight matrix of shape (n_h2, n_h1)
                    b2 -- bias vector of shape (n_h2, 1)
                    W3 -- weight matrix of shape (n_y, n_h2)
                    b3 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(0)
    
    parameters = {}
    L = len(layer_dims)   
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters
parameters = initialize_parameters(layer_dims)
# parameters
for i in parameters.keys():
    print("shape {} :".format(i), np.shape(parameters[i]))
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
A, W, b = dfX, parameters['W1'], parameters['b1']
Z, linear_cache = linear_forward(A, W, b)
print(Z.shape)
print("Z = " + str(Z))
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    
    assert(A.shape == Z.shape)
    
    cache = Z
    
    return A, cache
sigmoid(Z)[0]
def relu(Z):
    """
    Implement the RELU activation in numpy.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache
relu(Z)[0]
def softmax(Z):
    """
    Implement the softmax activation in numpy.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.exp(Z) / sum(np.exp(Z))
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache
softmax(Z)[0]
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "tanh" or "softmax"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    Z, linear_cache = linear_forward(A_prev, W, b)  
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        A, activation_cache = relu(Z)
        
    elif activation == "softmax":
        A, activation_cache = softmax(Z)   
        
    elif activation == "tanh":
        A, activation_cache = np.tanh(Z), Z
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)  

    return A, cache     # g(Z), ((A, W, b), g(Z))
A_prev, W, b = dfX, parameters['W1'], parameters['b1']
linear_activation_forward(A_prev, W, b, activation='tanh')[0]
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the LINEAR -> RELU -> LINEAR  TANH -> LINEAR -> SIGMOID -> LINEAR -> SOFTMAX computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2
    
#     for l, act in zip(range(1, L), ['relu', 'tanh', 'sigmoid']):
    for l in range(1, L):
        A_prev = A 
#         A, cache = linear_activation_forward(A_prev, parameters['W{}'.format(l)], parameters['b{}'.format(l)], activation = act)
        A, cache = linear_activation_forward(A_prev, parameters['W{}'.format(l)], parameters['b{}'.format(l)], activation = 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W{}'.format(L)], parameters['b{}'.format(L)], activation = "softmax")
    caches.append(cache)
    
    assert(AL.shape == (10, X.shape[1]))
            
    return AL, caches    # A[4], ((A[0], W1, b1, (Z1)), (A[1], W2, b2, (Z2)), (A[2], W3, b3, (Z3)), (A[3], W4, b4, (Z4)))
A4, caches = L_model_forward(dfX, parameters)
print(A4.shape)
A4
def compute_cost(AL, Y):
    """
    Implement the cost function.

    Arguments:
    AL -- probability matrix, output of the forward propagation (L_model_forward())
    Y -- ture"label" numpy ndarray
    
    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = -1/m * np.sum(np.sum(Y * np.log(AL), axis = 1, keepdims=True))
    
    assert(cost.shape == ())
    
    return cost
cost = compute_cost(A4, dfY)
cost
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ,A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a single Softmax unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    p, _ = softmax(Z)
    dZ = dA * p * (1 - p)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s, _ = sigmoid(Z)
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single Tanh unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    t = np.tanh(Z)
    dZ = dA * (1 - t) * (1 + t)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" or "tanh" or "softmax"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
dA_prev, dW, db = linear_activation_backward(dA=A4 - dfY, cache=caches[3], activation='softmax')
# dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the LINEAR -> RELU -> LINEAR  TANH -> LINEAR -> SIGMOID -> LINEAR -> SOFTMAX group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- ture"label" numpy ndarray
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)         
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # Initializing the backpropagation
    dAL = AL - dfY
    
    current_cache = caches[L-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    
#     for l, act_func in zip(reversed(range(L-1)), ['sigmoid', 'tanh', 'relu']):
    for l in reversed(range(L-1)):
        current_cache = caches[l]
#         dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l + 1)], current_cache, activation = act_func)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l + 1)], current_cache, activation = 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
grads = L_model_backward(A4, dfY, caches)
# grads
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]
        
    return parameters
parameters = update_parameters(parameters, grads, learning_rate=0.02)
# parameters
def L_layer_model(dfX, dfY, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: LINEAR -> RELU -> LINEAR  TANH -> LINEAR -> SIGMOID -> LINEAR -> SOFTMAX.
    
    Arguments:
    X -- data, numpy ndarray of shape (num_px * num_px, number of examples)
    Y -- ture"label" numpy ndarray (containing 0, 1), of shape (10, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []
    
    layers_dims = layer_sizes(dfX, dfY)
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(dfX, parameters)
        
        cost = compute_cost(AL, dfY)
        
        grads = L_model_backward(AL, dfY, caches)

        parameters = update_parameters(parameters, grads, learning_rate)
                
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
# %%time
# parameters = L_layer_model(dfX, dfY, learning_rate = 0.01, num_iterations = 5000, print_cost=True)
# parameters
# test, _ = L_model_forward(df_test, parameters)
# test = pd.DataFrame(pd.DataFrame(test).idxmax(), columns=['Label'])
# test.index.name='ImageId'
# test.index += 1 
# test
# test.tail()
# test.to_csv('test.csv')
