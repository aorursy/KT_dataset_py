!pip install --upgrade pip
!pip install arff
import os
import scipy as sp
import numpy as np
import arff
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# The main directory of the arff files with the .../phone/accel, .../phone/gyro, .../watch/accel, .../watch/gyro subdirectories
main_directory ='/kaggle/input/smartphone-and-smartwatch-activity-and-biometrics/wisdm-dataset/wisdm-dataset/arff_files'

def arff_to_array(arff_filepath):
    """
    Returns the numpy array equivalent of the arff files
    Parameters:
        arff_filepath (str): The file path of the arff file
    Returns:
        arr (numpy array): The numpy array equivalent of the arff file
    """
    file = arff.load(arff_filepath)
    lst = list(file)
    arr = np.asarray(lst)
    return arr

def list_file_directory(file_directory):
    """
    Returns a python list of all '.arff' files in the given file directory
    Parameters:
        file_directory (str): The file directory with the arff files
    Returns:
        arff_filepaths (list): A python list of all '.arff' files in the given file directory
    """
    arff_filepaths = []
    for root,dirs,files in os.walk(file_directory):
        for file in files:
            if file.endswith(".arff"):
                arff_filepaths. append(os.path.join(root,file))
    return arff_filepaths

def load_data():
    """
    Returns a numpy arrays of the '.arff' files in each of the four directoies
    Returns:
        raw_phone_accel(numpy array): A numpy array of the '.arff' files in the ".../phone/accel" directory
        raw_phone_gyro(numpy array): A numpy array of the '.arff' files in the ".../phone/gyro" directory
        raw_watch_accel(numpy array): A numpy array of the '.arff' files in the ".../watch/accel" directory
        raw_watch_gyro(numpy array): A numpy array of the '.arff' files in the ".../watch/gyro" directory
    """
    raw_phone_accel = np.empty((0,93))
    file_directory_phone_accel = main_directory + "/phone/accel"
    arff_filepaths_phone_accel = list_file_directory(file_directory_phone_accel)
    for i in range(len(arff_filepaths_phone_accel)):
        filepath = arff_filepaths_phone_accel[i]
        arr = arff_to_array(filepath)
        raw_phone_accel = np.append(raw_phone_accel, arr, axis=0)

    raw_phone_gyro = np.empty((0,93))
    file_directory_phone_gyro = main_directory + "/phone/gyro"
    arff_filepaths_phone_gyro = list_file_directory(file_directory_phone_gyro)
    for i in range(len(arff_filepaths_phone_gyro)):
        filepath = arff_filepaths_phone_gyro[i]
        arr = arff_to_array(filepath)
        raw_phone_gyro = np.append(raw_phone_gyro, arr, axis=0)

    raw_watch_accel = np.empty((0,93))
    file_directory_watch_accel = main_directory + "/watch/accel"
    arff_filepaths_watch_accel = list_file_directory(file_directory_watch_accel)
    for i in range(len(arff_filepaths_watch_accel)):
        filepath = arff_filepaths_watch_accel[i]
        arr = arff_to_array(filepath)
        raw_watch_accel = np.append(raw_watch_accel, arr, axis=0)

    raw_watch_gyro = np.empty((0,93))
    file_directory_watch_gyro = main_directory + "/watch/gyro"
    arff_filepaths_watch_gyro = list_file_directory(file_directory_watch_gyro)
    for i in range(len(arff_filepaths_watch_gyro)):
        filepath = arff_filepaths_watch_gyro[i]
        arr = arff_to_array(filepath)
        raw_watch_gyro = np.append(raw_watch_gyro, arr, axis=0)

    return raw_phone_accel, raw_phone_gyro, raw_watch_accel, raw_watch_gyro


""""
# Testing list_file_directory:
path = main_directory + "/phone/accel/data_1600_accel_phone.arff"
x = arff_to_array(path)
print(x)
print(x.shape)

# Testing list_file_directory:
filedir = main_directory + "/phone/gyro"
filepaths = list_file_directory(filedir)
print(filepaths[1])
print(len(filepaths))

# Testing arff_to_array using the output from list_file_directory:
filedir = main_directory + "/phone/gyro"
filepaths = list_file_directory(filedir)
filepath = filepaths[1]
array = arff_to_array(filepath)
print(array)
print("The size of the array is "  + str(array.shape))

# Testing list_file_directory:
phone_accel, phone_gyro, watch_accel, watch_gyro = load_data()
print(phone_accel)
print(phone_accel.shape) # should be 23074 x 93
print(phone_gyro)
print(phone_gyro.shape) # should be 17281 x 93
print(watch_accel)
print(watch_accel.shape) # should be 18211 x 93
print(watch_gyro)
print(watch_gyro.shape) # should be 16533 x 93
"""
def preprocessing(np_array):
    """
    Returns the unlabelled data (standardised)
    Parameters:
        np_array (numpy array): The numpy array to be standardised
    Returns:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
    """
    # Remove the first column (label) and the last columns(subject-id) of the array
    np_array = np.delete(np_array,[0,np_array.shape[1] - 1],1)
    # Convert from  'numpy.str_' numpy array to  a 'numpy.float'numpy array
    np_array = np_array.astype(np.float)
    # Standardization
    X = (np_array - np_array.mean()) / np_array.std()
    return X

def Kmeans_initialise_centroids(X, k):
    """
    Returns the initial centroids
    Parameters:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
        k (int): the number of clusters
    Returns:
        centroids (numpy array): A numpy array representing the intital centroids
    """
    # Shuffles the row indices of X
    random_idx = np.random.permutation(X.shape[0])
    # Indexes the first k elements of the random_idx array
    centroids = X[random_idx[:k]]
    return centroids

def compute_distance(X, centroids, k):
    """
    Computes the sum of the squared distance between data points and all centroids
    Parameters:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
        k (int): the number of clusters
        centroids (numpy array): A numpy array reporesenting the centroids
    Returns:
        distance (numpy array): A numpy array where each row represents
        a single data point and each column is the squared distance of that data
        point from each of centroid.
    """
    distance = np.zeros((X.shape[0], k))
    for k in range(k):
        # Calculates the Euclidean distance from each data point to each centroid
        row_norm = norm(X - centroids[k, :], axis=1)
        # Squares the Euclidean distance
        distance[:, k] = np.square(row_norm)
    return distance

def find_closest_centroid(distance):
    """
    Computes the index of the closest centroid
    Parameters:
        distance (numpy array): A numpy array where each row represents
        a single data point and each column is the squared distance of that data
        point from each centroid.
    Returns:
        label (numpy array): A row vetor where each row represents a signle
        data point and the value is the index of the cluster with the lowest
        squared distance from the data point
    """
    return np.argmin(distance, axis=1)

def compute_centroids(X, labels, k):
    """
    Computes the updated centroid
    Parameters:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
        label (numpy array): A row vetor where each row represents a signle
        data point and the value is the index of the cluster with the lowest
        squared distance from the data point
        k(int): the number of clusters
    Returns:
        centroids (numpy array): A numpy array representing the updated centroids
    """
    # Initalise centroids as an numpy array of shape (k, n)
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        # Select all data points that belong to cluster i and compute
        # the mean of these data points (each feature individually)
        # this will be our new cluster centroids
        centroids[i, :] = np.mean(X[labels == i, :], axis=0)
    return centroids

def Kmeans_clustering(X, k, max_iter):
    """
    Implements K-means clustering 
    Parameters:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
        k (int): the number of clusters
        max_iter (int): The maximum number of iterations
    Returns:
        centroids (numpy array): A numpy array representing the centroids
    """
    # Initalise the centroids
    centroids = Kmeans_initialise_centroids(X,k)
    for i in range(max_iter):
        old_centroids = centroids
        # Compute the sum of the squared distance between data points
        # and all centroids
        distance = compute_distance(X, old_centroids, k)
        labels = find_closest_centroid(distance)
        # Assign each data point to the closest cluster (centroid)
        centroids = compute_centroids(X,labels,k)
        # If the updated centroid is still the same, then the algorithm has converged
        if np.all(old_centroids == centroids):
            print("The maximum number of iterations is actually " + str(i))
            break
    return labels, centroids


def compute_sse(X, labels, centroids):
    """
    Computes the sum of squared error
    Parameters:
        X (numpy array): The unlabelled data points as a numpy array of shape
        (m,n) containing m data points (observations) each of dimension n
        label (numpy array): A row vetor where each row represents a signle
        data point and the value is the index of the cluster with the lowest
        squared distance from the data point
        centroids (numpy array): A numpy array reporesenting the centroids
    Returns:
        sse (int): The sum of squared error
    """
    distance = np.zeros(X.shape[0])
    k = centroids.shape[0]
    for i in range(k):
        # Calculates the Euclidean distance between the data point and
        # the cluster's centroid
        distance[labels == i] = norm(X[labels == i] - centroids[i], axis=1)
    return np.sum(np.square(distance))


# K-MEANS CLUSTERING
phone_accel, phone_gyro, watch_accel, watch_gyro = load_data()
X = preprocessing(watch_gyro) 
k = 3 # number of clusters
max_iter = 50
labels, centroids = Kmeans_clustering(X, k, max_iter)
sse = compute_sse(X, labels, centroids)
print("The sum of squared error is " + str(sse))

pca = PCA(n_components=2)
pca.fit(X)
pca_result = pca.transform(X)
pca_one = pca_result[:,0]
pca_two = pca_result[:,1]

pca_centroid = pca.transform(centroids)
pca_centroid_one = pca_centroid[:,0]
pca_centroid_two = pca_centroid[:,1]

plt.scatter(pca_one, pca_two, c = labels, cmap= plt.cm.Paired)
plt.scatter(pca_centroid_one, pca_centroid_two, marker='*')
plt.xlabel('pca axis 1')
plt.ylabel('pca axis 2')
plt.show()
def preprocessing(np_array):
    """
    Returns the examples and their corresponding lables
    Parameters:
        np_array (numpy array): The raw numpy array 
    Returns:
        X (numpy array): The data points as a numpy array of shape 
        (m,n) containing m data points (observations) each of dimension n
        Y (numpy array): The labels as a numpy array of shape (m, 1) containing the 
        label for each data point
    """
    # Remove the last columns(subject-id) of the array
    np_array = np.delete(np_array,[np_array.shape[1] - 1],1)
    # Get the examples
    examples = np_array[:,1:]
    examples = examples.astype(np.float)
    # Standardization
    X = (examples - examples.mean())/(examples.std())
    # Get the labels
    raw_labels = np_array[:,0]
    raw_labels_as_lists = raw_labels.tolist()
    labels_as_lists = [ord(letter) - 65 for letter in raw_labels_as_lists]
    labels_as_array = np.asarray(labels_as_lists, dtype = float)
    # One-hot labelling
    one_hot_labels = np.zeros((X.shape[0],19))
    for i in range(X.shape[0]):
        one_hot_labels[i,int(labels_as_array[i])] = 1
    return X, one_hot_labels

def sigmoid(Z):
    """
    Implements the sigmoid activation of Z
    Parameters:
        Z (numpy array): The raw numpy array 
    Return: 
        A (numpy array): The output of sigmoid(Z), same shape as Z
        cache (numpy array): The original input Z, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Implements the relu activation of Z
    Parameters:
        Z (numpy array): The raw numpy array 
    Return: 
        A (numpy array): The output of relu(Z), same shape as Z
        cache (numpy array): The original input Z, useful during backpropagation
    """
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def tanh(Z):
    """
    Implements the tanh activation of Z
    Parameters:
        Z (numpy array): The raw numpy array 
    Return: 
        A (numpy array): The output of tanh(Z), same shape as Z
        cache (numpy array): The original input Z, useful during backpropagation
    """
    A = np.tanh(Z)
    cache = Z
    return A, cache

def softmax(Z):
    """
    Implements the softmax activation of Z
    Parameters:
        Z (numpy array): The raw numpy array 
    Return: 
        A (numpy array): The output of softmax(Z), same shape as Z
        cache (numpy array): The original input Z, useful during backpropagation
    """
    expZ = np.exp(Z - np.max(Z))
    A = expZ / expZ.sum(axis=0, keepdims=True)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implements the gradient (also called the slope or derivative) of the
    sigmoid function with respect to its input Z.
    Parameters: 
        dA (numpy array): post-activation gradient
        cache (numpy array): 'Z' where we store for computing backward propagation efficiently
    Return: 
        dZ(numpy array): Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    """
    Implements the gradient (also called the slope or derivative) of the
    relu function with respect to its input Z.
    Parameters: 
        dA (numpy array): post-activation gradient
        cache (numpy array): 'Z' where we store for computing backward propagation efficiently
    Return: 
        dZ(numpy array): Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def tanh_backward(dA, cache):
    """
    Implements the gradient (also called the slope or derivative) of the
    tanh function with respect to its input Z.
    Parameters: 
        dA (numpy array): post-activation gradient
        cache (numpy array): 'Z' where we store for computing backward propagation efficiently
    Return: 
        dZ(numpy array): Gradient of the cost with respect to Z
    """
    Z = cache
    a, cache = tanh(Z)
    dZ = dA * (1-a**2)
    return dZ

def initialize_parameters_deep(layer_dims):
    """
    Returns the parameters "W1", "b1", ..., "WL", "bL"
    Parameters:
        layer_dims (python list): contains the dimensions of each layer in our network
    
    Returns:
        parameters (python dictionary): contains the parameters "W1", "b1", ..., "WL", "bL":
            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
            bl -- bias vector of shape (layer_dims[l], 1)
    """  
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / (layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.
    Parameters:
        A (numpy array): activations from previous layer (or input data): (size of previous layer, number of examples)
        W (numpy array): weights matrix of shape (size of current layer, size of previous layer)
        b (numpy array): bias vector of shape (size of the current layer, 1)
    Returns:
        Z (numpy array): the input of the activation function, also called pre-activation parameter 
        cache (python dictionary): contains "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer
    Parameters:
        A_prev (numpy array): activations from previous layer (or input data): (size of previous layer, number of examples)
        W (numpy array): weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b (numpy array): bias vector, numpy array of shape (size of the current layer, 1)
        activation (string): the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "tanh", or "softmax"
    Returns:
        A (numpy array):  the output of the activation function, also called the post-activation value 
        cache (python dictionary): contains "linear_cache" and "activation_cache"; stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z) # caches Z
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, activation):
    """
    Implements forward propagation for the [LINEAR->ACTIVATION]*(L-1)->LINEAR->SOFTMAX computation
    Parameters: 
        X (numpy array): data of shape (input size, number of examples)
        parameters (python dictionary): output of initialize_parameters_deep()
        activation (string): the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "tanh", or "softmax"
    Returns:
        AL (numpy array):  last post-activation value
        caches (python list): list of caches containing:
            every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
            the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    activation = activation.lower()
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='softmax')
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, parameters, lambd):
    """
    Implements the cross entrophy cost function
    Arguments:
        AL (numpy array): probability vector corresponding to your label predictions
        Y (numpy array): true "label" vector 
        parameters (python dictionary): output of the forward propagation
        lambd (int): the regularization rate
    Returns:
        cost (int):  categorical cross-entropy cost
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    L2_regularization_cost = 0
    for l in range(1, L):
        L2_regularization_cost = L2_regularization_cost + lambd*(np.sum(np.square(parameters['W' + str(l)])))
    cost = -1/m * np.sum(Y * np.log(AL + 1e-8)) + L2_regularization_cost
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    Parameters: 
        dZ (numpy array): gradient of the cost with respect to the linear output (of current layer l)
        cache (tuples): tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns:
        dA_prev (numpy array): gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (numpy array): gradient of the cost with respect to W (current layer l), same shape as W
        db (numpy array): gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
    PArameters:
        dA (numpy array): post-activation gradient for current layer l 
        cache (tuples): tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation (string): the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "tanh", or "softmax"
    Returns:
        dA_prev (numpy array): gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (numpy array): gradient of the cost with respect to W (current layer l), same shape as W
        db (numpy array): gradient of the cost with respect to b (current layer l), same shape as b
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
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activation, lambd):
    """
    Implement the backward propagation for the [LINEAR->ACTIVATION] * (L-1) -> LINEAR -> SOFTMAX group  
    Parameters:
        AL (numpy array): probability vector, output of the forward propagation (L_model_forward())
        Y (numpy array): true "label" vector (containing 0 if non-cat, 1 if cat)
        caches (python list): list of caches containing:
                every cache of linear_activation_forward() with ACTIVATION (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "softmax" (it's caches[L-1])
        activation (string): the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "tanh", or "softmax"
        lambd (int): the regularization rate
    Returns:
        grads (dictionary): A dictionary with the gradients
            grads["dA" + str(l)] = ... 
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dZ = AL - Y
    current_cache = caches[L - 1]
    linear_cache, activation_cache = current_cache
    A, WL, b = linear_cache
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache)
    grads["dW" + str(L)] = grads["dW" + str(L)] + (lambd * WL)/m

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation)
        linear_cache, activation_cache = current_cache
        A, WL, b = linear_cache # Get WL for regularisation
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp + (lambd * WL)/m
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent 
    Parameters:
        parameters (python dictionary): containing your parameters 
        grads (python dictionary): containing your gradients, output of L_model_backward
    Returns:
        parameters -- python dictionary containing your updated parameters 
            parameters["W" + str(l)] = ... 
            parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    Parameters:
        X (numpy array): input data, of shape (input size, number of examples)
        Y (numpy array): true "label" vector 
        mini_batch_size (int): size of the mini-batches, integer 
    Returns:
        mini_batches (python list): list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def initialize_adam(parameters) :
    """"
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Parameters:
        parameters (python dictionary): contains the parameters.
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    Returns: 
        v (python dictionary): contains the exponentially weighted average of the gradient.
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        s (python dictionary): contains the exponentially weighted average of the squared gradient.
            s["dW" + str(l)] = ...
            s["db" + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return v, s

def update_parameters_with_adam(parameters, grads, learning_rate, v, s, t, beta1, beta2,  epsilon):
    """
    Update parameters using Adam
    Parameters: 
        parameters (python dictionary): contains your parameters:
        grads (python dictionary): contains your gradients for each parameters           
        v (python dictionary): Adam variable, moving average of the first gradient
        s (python dictionary): Adam variable, moving average of the squared gradient
        learning_rate (int): the learning rate, scalar.
        beta1 (int): Exponential decay hyperparameter for the first moment estimates 
        beta2 (int): Exponential decay hyperparameter for the second moment estimates 
        epsilon (int): hyperparameter preventing division by zero in Adam updates
    Returns:
        parameters (python dictionary): contains your updated parameters 
        v (python dictionary): Adam variable, moving average of the first gradient
        s (python dictionary): Adam variable, moving average of the squared gradient
    """
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)
    return parameters, v, s

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, lambd, activation, mini_batch_size, optimizer, optimizer_hyperameters):
    """
    Implements a L-layer neural network: [LINEAR->ACTIVATION]*(L-1)->LINEAR->SOFTMAX.
    Parameters:
        X (numpy array): data
        Y (numpy array): true "label" vector 
        layers_dims (python list): contains the input size and each layer size, of length (number of layers + 1).
        learning_rate (int): learning rate of the gradient descent update rule
        num_iterations (int): number of iterations of the optimization loop
        lambd (int): the regularization rate
        activation (string): the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "tanh", or "softmax"
        mini_batch_size (int): size of the mini-batches, integer 
        optimizer (string): can be an empty string or "adam" which calls for Adams optimization
        optimizer_hyperparameters (tuples): tuple caoninting the hyperparameters for Adams optimization
    Returns:
        parameters (python dictionary):  parameters learnt by the model. They can then be used to predict.
    """
    costs = []
    activation = activation.lower()
    optimizer = optimizer.lower()
    parameters = initialize_parameters_deep(layers_dims)
    if (optimizer == "adam"):
        v, s = initialize_adam(parameters)

    for i in range(0, num_iterations):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            AL,caches = L_model_forward(minibatch_X, parameters, activation)
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)
            grads = L_model_backward(AL, minibatch_Y, caches, activation, lambd)
            if (optimizer == "adam"):
                (t, beta1, beta2, epsilon) = optimizer_hyperameters
                parameters, v, s = update_parameters_with_adam(parameters, grads, learning_rate, v, s, t, beta1, beta2,  epsilon)
            else:
                parameters = update_parameters(parameters, grads, learning_rate)   
        if (i % 10 == 0):
            costs.append(cost)
            print("Cost: " + str(cost))
    # plt.plot(np.squeeze(costs))
    train_accuracy = predict(X, Y,parameters, activation)
    print("Train Accuracy: " + str(train_accuracy))
    return parameters

def predict(X, Y, parameters, activation):
    AL, caches = L_model_forward(X, parameters, activation)
    y_hat = np.argmax(AL, axis = 0)
    Y = np.argmax(Y, axis = 0)
    accuracy = (y_hat == Y).mean()
    return accuracy * 100

# NEURAL NET 
phone_accel, phone_gyro, watch_accel, watch_gyro = load_data()
X,Y = preprocessing(watch_gyro)
X = np.transpose(X)
Y = np.transpose(Y)

s = math.floor((2/3) * X.shape[1]) # Pick two-thirds of the sample to pick for training
idx = np.random.choice(X.shape[1],size= s, replace=False)
idx_train = idx.tolist()
X_train = X[:, idx_train]
Y_train = Y[:, idx_train]
# print(X_train)
# print(X_train.shape)
idx = np.arange(X.shape[1]).tolist()
idx_test = list(set(idx).difference(idx_train))
X_test = X[:, idx_test]
Y_test = Y[:, idx_test]
# print(X_test)
# print(X_test.shape)
features = X.shape[0]
classes = Y.shape[0]

layers_dims = [features, 15, classes]
learning_rate = 0.3
num_iterations = 15
lambd = 0 # non-regularised is lambd is zero
activation = "sigmoid" # can be "sigmoid", "relu", "tanh"
mini_batch_size = 8
optimizer = "" # can be "adam", "" (empty string)
# Adam optimizer hyperparameters
t = 2
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
optimizer_hyperameters = (t, beta1, beta2, epsilon)

parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate, num_iterations, lambd, activation, mini_batch_size, optimizer, optimizer_hyperameters)
test_accuracy = predict(X_test, Y_test,parameters, activation)
print("Test Accuracy: " + str(test_accuracy))