import numpy as np

import matplotlib.pyplot as plt

import math

import h5py

import scipy

from PIL import Image

from scipy import ndimage

from h5py_utils import *

import sklearn

from sklearn.utils import shuffle



%matplotlib inline
# Loading the data (cat/non-cat)

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
# Example of a picture

index = 500

plt.imshow(train_x_orig[index])

print ("y = " + str(train_y[:, index]) + ", it's a '" + classes[np.squeeze(train_y[:, index])] +  "' apple.")
### START CODE HERE ### (≈ 3 lines of code)

m_train = train_x_orig.shape[0]

m_test = test_x_orig.shape[0]

num_px = train_x_orig.shape[1]

### END CODE HERE ###



print ("Number of training examples: m_train = " + str(m_train))

print ("Number of testing examples: m_test = " + str(m_test))

print ("Height/Width of each image: num_px = " + str(num_px))

print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")

print ("train_set_x shape: " + str(train_x_orig.shape))

print ("train_set_y shape: " + str(train_y.shape))

print ("test_set_x shape: " + str(test_x_orig.shape))

print ("test_set_y shape: " + str(test_y.shape))
# Reshape the training and test examples and then "normalize" by dividing by 255



### START CODE HERE ### (≈ 2 lines of code)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T

test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.

test_x = test_x_flatten/255.



### END CODE HERE ###



print ("train_set_x_flatten shape: " + str(train_x_flatten.shape))

print ("train_set_y shape: " + str(train_y.shape))

print ("test_set_x_flatten shape: " + str(test_x_flatten.shape))

print ("test_set_y shape: " + str(test_y.shape))

print ("sanity check after reshaping: " + str(train_x_flatten[0:5,0]))
def random_mini_batches2(X, Y, available_indices, mini_batch_size = 1024, seed = 0, start = False):

    np.random.seed(seed)            # To make your "random" minibatches the same as ours

    m = X.shape[1]                  # number of training examples

    

    if(start):

        available_indices = np.array([i for i in range(train_x.shape[1])])

        np.random.shuffle(available_indices)

    

    if available_indices.shape[0] >= mini_batch_size:

        current_minibatch = available_indices[0:mini_batch_size]

        available_indices = available_indices[mini_batch_size:-1]

    else:

        current_minibatch = available_indices

        available_indices = [-1]

        

    mini_batch_x = X[:,current_minibatch]

    mini_batch_y = Y[:,current_minibatch]

            

    return mini_batch_x, mini_batch_y, available_indices
def initialize_parameters_he(layers_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the size of each layer.

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])

                    b1 -- bias vector of shape (layers_dims[1], 1)

                    ...

                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])

                    bL -- bias vector of shape (layers_dims[L], 1)

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layers_dims) - 1 # integer representing the number of layers

     

    for l in range(1, L + 1):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])

        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        ### END CODE HERE ###

        

        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

        

    return parameters
parameters = initialize_parameters_he([2, 4, 1])

print("W1 = " + str(parameters["W1"]))

print("Variance of W1 = " + str(np.var(parameters["W1"][0])) + " 1/n = " + str(1/2))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("Variance of W2 = " + str(np.var(parameters["W2"][0])) + " 1/n = " + str(1/4))

print("b2 = " + str(parameters["b2"]))
def linear_forward(A, W, b):

    """

    Implement the linear part of a layer's forward propagation.



    Arguments:

    A -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)



    Returns:

    Z -- the input of the activation function, also called pre-activation parameter 

    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    

    ### START CODE HERE ### (≈ 1 line of code)

    Z = None

    ### END CODE HERE ###

    

    assert(Z.shape == (W.shape[0], A.shape[1]))

        

    return Z
def linear_activation_forward(A_prev, W, b, activation):

    """

    Implement the forward propagation for the LINEAR->ACTIVATION layer



    Arguments:

    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"



    Returns:

    A -- the output of the activation function, also called the post-activation value 

    cache -- a python tuple containing "linear_cache" and "activation_cache";

             stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        ### START CODE HERE ### (≈ 2 lines of code)

        Z = linear_forward(A_prev, W, b)

        A = None

        ### END CODE HERE ###

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        ### START CODE HERE ### (≈ 2 lines of code)

        Z = linear_forward(A_prev, W, b)

        A = None

        ### END CODE HERE ###

    

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    

    ### IMPORTANT!! WE STORE THE VARIABLES OF FORWARD PROPOGATION HERE FOR LATER USE IN BACKWARD PROPOGATION ###

    cache = (Z, A_prev, W, b)



    return A, cache
def L_model_forward(X, parameters):

    """

    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    

    Arguments:

    X -- data, numpy array of shape (input size, number of examples)

    parameters -- output of initialize_parameters_deep()

    

    Returns:

    AL -- last post-activation value

    caches -- list of caches containing:

                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)

    """



    caches = []

    A = X

    L = len(parameters) // 2                  # number of layers in the neural network

    

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A 

        ### START CODE HERE ### (≈ 2 lines of code)

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")

        caches.append(cache)

        ### END CODE HERE ###

    

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    ### START CODE HERE ### (≈ 2 lines of code)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")

    caches.append(cache)

    ### END CODE HERE ###

    

    assert(AL.shape == (1,X.shape[1]))

            

    return AL, caches
def compute_cost(AL, Y):

    """

    Implement the cost function defined by equation (7).



    Arguments:

    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)

    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)



    Returns:

    cost -- cross-entropy cost

    """

    

    m = Y.shape[1]



    # Compute loss from aL and y.

    ### START CODE HERE ### (≈ 1 lines of code)

    cost = -(1/m)*(np.dot(Y, (np.log(AL)).T)+np.dot((1-Y), (np.log(1-AL)).T))

    ### END CODE HERE ###

    

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    assert(cost.shape == ())

    

    return cost
def linear_backward(dZ, cache, final = False):

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

    dW = (1/m)*np.dot(dZ, A_prev.T)

    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)



    if(final):

        #this is to avoid the last computational step that is quite CPU intensive and unnecessairy. However, this is not a true statement in a mathematical sense

        dA_prev = A_prev

    else:

        ### START CODE HERE ### (≈ 3 lines of code)

        dA_prev = np.dot(W.T, dZ)

        ### END CODE HERE ###



    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation, final = False):

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

    

    Z, A_prev, W, b = cache

    

    linear_cache = (A_prev, W, b)

    

    if activation == "relu":

        dZ = relu_backward(dA, Z)

        

        if(final):

            dA_prev, dW, db = linear_backward(dZ, linear_cache, final = True)

        else:

            ### START CODE HERE ### (≈ 2 lines of code)

            dA_prev, dW, db = linear_backward(dZ, linear_cache)

            ### END CODE HERE ###

        

    elif activation == "sigmoid":

        ### START CODE HERE ### (≈ 2 lines of code)

        dZ = sigmoid_backward(dA, Z)

        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        ### END CODE HERE ###

    

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):

    """

    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    

    Arguments:

    AL -- probability vector, output of the forward propagation (L_model_forward())

    Y -- true "label" vector (containing 0 if rotten apple, 1 if good apple)

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

    L = len(caches) # the number of layers

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    

    # Initializing the backpropagation - this begins the backpropogation by calculating the derivative of the cost function with respect to the activations of the final output layer

    ### START CODE HERE ### (1 line of code)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    ### END CODE HERE ###

    

    

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    ### START CODE HERE ### (approx. 2 lines)

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    ### END CODE HERE ###

    

    

    # Loop from l=L-2 to l=0

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        ### START CODE HERE ### (approx. 5 lines)

        current_cache = caches[l]

        if l != 0:

            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")

        ### END CODE HERE ###

        else:

            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu", final = True)

        

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

            



    return grads
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

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    # We use l+1 here because l starts at 0 but W1 & b1 are the first weight parameters

    ### START CODE HERE ### (≈ 3 lines of code)

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]

    ### END CODE HERE ###

    return parameters
def initialize_adam(parameters) :

    """

    Initializes v and s as two python dictionaries with:

                - keys: "dW1", "db1", ..., "dWL", "dbL" 

                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    

    Arguments:

    parameters -- python dictionary containing your parameters.

                    parameters["W" + str(l)] = Wl

                    parameters["b" + str(l)] = bl

    

    Returns: 

    v -- python dictionary that will contain the exponentially weighted average of the gradient.

                    v["dW" + str(l)] = ...

                    v["db" + str(l)] = ...

    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.

                    s["dW" + str(l)] = ...

                    s["db" + str(l)] = ...



    """

    

    L = len(parameters) // 2 # number of layers in the neural networks

    v = {}

    s = {}

    

    # Initialize v, s. Input: "parameters". Outputs: "v, s".

    for l in range(L):

    ### START CODE HERE ### (approx. 4 lines)

        v["dW" + str(l+1)] = np.zeros(((parameters['W' + str(l+1)]).shape[0], (parameters['W' + str(l+1)]).shape[1]))

        v["db" + str(l+1)] = np.zeros(((parameters['b' + str(l+1)]).shape[0], (parameters['b' + str(l+1)]).shape[1]))

        s["dW" + str(l+1)] = np.zeros(((parameters['W' + str(l+1)]).shape[0], (parameters['W' + str(l+1)]).shape[1]))

        s["db" + str(l+1)] = np.zeros(((parameters['b' + str(l+1)]).shape[0], (parameters['b' + str(l+1)]).shape[1]))

    ### END CODE HERE ###

    

    return v, s
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,

                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    """

    Update parameters using Adam

    

    Arguments:

    parameters -- python dictionary containing your parameters:

                    parameters['W' + str(l)] = Wl

                    parameters['b' + str(l)] = bl

    grads -- python dictionary containing your gradients for each parameters:

                    grads['dW' + str(l)] = dWl

                    grads['db' + str(l)] = dbl

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    learning_rate -- the learning rate, scalar.

    beta1 -- Exponential decay hyperparameter for the first moment estimates 

    beta2 -- Exponential decay hyperparameter for the second moment estimates 

    epsilon -- hyperparameter preventing division by zero in Adam updates



    Returns:

    parameters -- python dictionary containing your updated parameters 

    v -- Adam variable, moving average of the first gradient, python dictionary

    s -- Adam variable, moving average of the squared gradient, python dictionary

    """

    

    L = len(parameters) // 2                 # number of layers in the neural networks

    v_corrected = {}                         # Initializing first moment estimate, python dictionary

    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    

    # Perform Adam update on all parameters

    for l in range(L):

        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".

        ### START CODE HERE ### (approx. 2 lines)

        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)]+(1-beta1)*grads['dW' + str(l+1)]

        v["db" + str(l+1)] = beta1*v["db" + str(l+1)]+(1-beta1)*grads['db' + str(l+1)]

        ### END CODE HERE ###



        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".

        ### START CODE HERE ### (approx. 2 lines)

        v_corrected["dW" + str(l+1)] =  (v["dW" + str(l+1)])/(1-np.power(beta1, t))

        v_corrected["db" + str(l+1)] = (v["db" + str(l+1)])/(1-np.power(beta1, t))

        ### END CODE HERE ###



        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".

        ### START CODE HERE ### (approx. 2 lines)

        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)]+(1-beta2)*grads['dW' + str(l+1)]**2

        s["db" + str(l+1)] = beta2*s["db" + str(l+1)]+(1-beta2)*grads['db' + str(l+1)]**2

        ### END CODE HERE ###



        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".

        ### START CODE HERE ### (approx. 2 lines)

        s_corrected["dW" + str(l+1)] = (s["dW" + str(l+1)])/(1-np.power(beta2, t))

        s_corrected["db" + str(l+1)] = (s["db" + str(l+1)])/(1-np.power(beta2, t))

        ### END CODE HERE ###



        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".

        ### START CODE HERE ### (approx. 2 lines)

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*(v_corrected["dW" + str(l+1)])/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*(v_corrected["db" + str(l+1)])/(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

        ### END CODE HERE ###



    return parameters, v, s
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    

    np.random.seed(1)

    L = len(layers_dims)             # number of layers in the neural networks

    costs = []                       # to keep track of the cost

    

    # Parameters initialization. (≈ 1 line of code)

    parameters = initialize_parameters_he(layers_dims)

    



    # Loop (gradient descent) on mini-batches

    for i in range(0, num_iterations):    



        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

        ### START CODE HERE ### (≈ 1 line of code)

        AL, caches = L_model_forward(X, parameters)

        ### END CODE HERE ###



        # Compute cost.

        ### START CODE HERE ### (≈ 1 line of code)

        cost = compute_cost(AL, Y)

        ### END CODE HERE ###



        # Backward propagation.

        ### START CODE HERE ### (≈ 1 line of code)

        grads = L_model_backward(AL, Y, caches)

        ### END CODE HERE ###



        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

        

        freq_print = np.ceil(num_iterations/20)

        

        # Print the cost every 50 training example

        if print_cost and i % freq_print == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % freq_print == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
def L_layer_model_extended(X, Y, layers_dims, optimizer = "adam", learning_rate = 0.0007, mini_batch_size = 1024, beta = 0.9,

          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost=False):#lr was 0.009

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    L = len(layers_dims)             # number of layers in the neural networks

    costs = []                       # to keep track of the cost

    t = 0                            # initializing the counter required for Adam update

    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours

    

    # Parameters initialization. (≈ 1 line of code)

    parameters = initialize_parameters_he(layers_dims)

    

    

    #Initialize the optimizer

    if optimizer == "gd":

        pass # no initialization required for gradient descent

    elif optimizer == "adam":

        v, s = initialize_adam(parameters)

        

    AL, caches = L_model_forward(X, parameters)

    cost = compute_cost(AL, Y)

    print ("Initial Cost: " + str(cost))

    costs.append(cost)

    

    for i in range(num_epochs):

        

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch

        seed = seed + 1

        start_epoch = True

        

        minibatch_X, minibatch_Y, available_indices = random_mini_batches2(X, Y, [-1], mini_batch_size, seed = 0, start = True)

        

        j = 0

        

        while True:           

            

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.

            ### START CODE HERE ### (≈ 1 line of code)

            AL, caches = L_model_forward(minibatch_X, parameters)

            ### END CODE HERE ###



            # Compute cost.

            ### START CODE HERE ### (≈ 1 line of code)

            cost = compute_cost(AL, minibatch_Y)

            ### END CODE HERE ###

            

            j = j + 1

            

            if print_cost and j % 10 == 0:

                print ("Cost after mini-batch %i: %f" %(j, cost))

                #costs.append(cost)

            

            # Backward propagation.

            ### START CODE HERE ### (≈ 1 line of code)

            grads = L_model_backward(AL, minibatch_Y, caches)

            ### END CODE HERE ###



            # Update parameters.

            ### START CODE HERE ### (≈ 1 line of code)

            if optimizer == "gd":

                parameters = update_parameters(parameters, grads, learning_rate)

            elif optimizer == "adam":

                t = t + 1

                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

            ### END CODE HERE ###

            

            if available_indices[0] < 0:

                break

            

            minibatch_x, minibatch_y, available_indices = random_mini_batches2(X, Y, available_indices, mini_batch_size, seed = 0, start = False)

            

        freq_print = np.ceil(num_epochs/20)

                       

        if print_cost and i % freq_print == 0:

            print ("Cost after epoch %i: %f" %(i, cost))

        if print_cost and i % freq_print == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters
n_x = 196608 #num_px * num_px * num_channels = 256*256*3 = 196,608 inputs

layers_dims = [n_x, 20, 7, 5, 1] #this is a 4-layer model. The layers have 20 hidden units, 7 hidden units, 5 hidden units, and 1 output node, respectively. There can only be one output node.
#parameters1 = L_layer_model(train_x, train_y, layers_dims, num_iterations = 500, print_cost = True)
#parameters2 = L_layer_model_extended(train_x, train_y, layers_dims, optimizer = "adam", learning_rate = 0.001, mini_batch_size = 512, beta = 0.9,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 500, print_cost=True)
def predict(X, y, parameters):

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

    probas, caches = L_model_forward(X, parameters)



    

    # convert probas to 0/1 predictions

    for i in range(0, probas.shape[1]):

        if probas[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    

    #print results

    #print ("predictions: " + str(p))

    #print ("true labels: " + str(y))

    print("Accuracy: "  + str(np.sum((p == y)/m)))

        

    return p
#pred_train = predict(train_x, train_y, parameters1)
#pred_test = predict(test_x, test_y, parameters1)
#pred_train = predict(train_x, train_y, parameters2)
#pred_test = predict(test_x, test_y, parameters2)
def print_mislabeled_images(classes, X, y, p):

    """

    Plots images where predictions and truth were different.

    X -- dataset

    y -- true labels

    p -- predictions

    """

    a = p + y

    mislabeled_indices = np.asarray(np.where(a == 1))

    plt.rcParams['figure.figsize'] = (500, 500) # set default size of plots

    num_images = len(mislabeled_indices[0])

    for i in range(10):

        index = mislabeled_indices[1][i]

        

        plt.subplot(2, num_images, i + 1)

        plt.imshow(X[:,index].reshape(256,256,3), interpolation='nearest')

        plt.axis('off')

        plt.title("Prediction: " + classes[int(p[0,index])] + " \n Class: " + classes[y[0,index]])

#print_mislabeled_images(classes, test_x, test_y, pred_test)
"""

my_image = "my_image.jpg" # change this to the name of your image file 

my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)



fname = "images/" + my_image

image = np.array(ndimage.imread(fname, flatten=False))

my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))

my_image = my_image/255.

my_predicted_image = predict(my_image, my_label_y, parameters)



plt.imshow(image)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

"""