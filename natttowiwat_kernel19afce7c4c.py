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
        
img = '/kaggle/input'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.image as mpimg
def rgb_to_grey(img):
    if len(img.shape) == 2: #if there are only two scales then it is grayscale image
        return(img)
    greyImage = np.zeros(img.shape)
    Red = img[:,:,0] * 0.299
    Green = img[:,:,1] * 0.587
    Blue = img[:,:,2] * 0.114
    
    greyImage = Red + Green + Blue
    
    return(greyImage)
    
def img_standardize(img, x_bin = 100, y_bin = 100):
    x_splits = np.linspace(0,img.shape[1] - 1, x_bins + 1, dtype = int)
    y_splits = np.linspace(0,img.shape[0] - 1, x_bins + 1, dtype = int)
    compressed = np.zeros((y_bins,x_bins))
    
    for i in range(y_bins):
        for j in range(x_bins):
            temp = np.mean(img[y_splits[i]:y_splits[i+1], x_splits[j]:x_splits[j+1]])
            if math.isnan(temp):
                if y_splits[i] == y_splits[i+1]:
                    compressed[i,j] = compressed[i-1,j]
                else:
                    compressed[i,j] = compressed[i,j-1]
            else: 
                compressed[i,j] = int(temp)
                
    return(compressed)

train_files_names = os.listdir(train_dir)[::5]
imgs_train = [rgb_to_grey(mpimg.imread(train_dir + '/' + file, format = 'JGP')) for file in os.listdir(train_dir)]



# Reshape the training and test examples 
train_x_flatten = imgs_train.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

def initialize_parameters(n_x, n_h, n_y, beta):
    W1 = np.random.randn(n_h,n_x) * beta
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y,n_h) * beta
    b2 = np.zeros((n_y, 1))
   
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    
def initialize_parameters_deep(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
  
    return parameters
def linear_forward(A, W, b):

    Z = np.dot(W,A) + b
   
    return Z, cache
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def tanh(z):
    s = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
    return s

def relu(z):
    s = np.maximum(0,z)
    return s

def softmax(z):
    expo = np.exp(z)
    expo_sum = np.sum(np.exp(z))
    return expo/expo_sum

def linear_activation_forward(A_prev, W, b, activation):
    #""""""
   # Arguments:
   # A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
   # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
   # b -- bias vector, numpy array of shape (size of the current layer, 1)
   # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

   # Returns:
   # A -- the output of the activation function, also called the post-activation value 
   # cache -- a python tuple containing "linear_cache" and "activation_cache";
   #          stored for computing the backward pass efficiently
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation == "hyptan":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
        
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    
    return A, cache
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
        
    for l in range(1, L//3):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "softmax")
        caches.append(cache)
        
    for l in range(L//3, 2*L//3):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "hyptan")
        caches.append(cache)
        
    for l in range(2*L//3, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "softmax")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
   
            
    return AL, caches
def compute_cost(AL, Y):
  
    m = Y.shape[1]


    logprobs = np.multiply(np.log(AL),Y) +  np.multiply(np.log(1-AL), (1-Y))
    cost = -1/m*np.sum(logprobs)

    cost = np.squeeze(cost)     
    
    return cost
def linear_backward(dZ, cache):
 
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m*np.dot(dZ, A_prev.T)
    db = 1./m*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "hyptan":
        dZ = hyptan_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. 
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
 
    for l in reversed(range(L-1)):
        # lth layer: (SOFTMAX -> LINEAR) gradients.
    
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],  current_cache, activation = "softmax")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. 
    parameters = initialize_parameters_deep(layers_dims)
   
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    
        AL, caches = L_model_forward(X, parameters)
      
        # Compute cost.
    
        cost = compute_cost(AL, Y)
       
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
   
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
def predict(parameters, X):
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X,parameters)
    predictions = (A2 > 0.5)
 
    return predictions