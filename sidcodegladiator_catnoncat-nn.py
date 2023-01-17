# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import h5py

import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
def load_data():

    train_dataset = h5py.File('/kaggle/input/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    

    test_dataset = h5py.File('/kaggle/input/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
def initialize_parameters(layers_dims_vector):

    no_of_layers = len(layers_dims_vector)

    parameters={}

    for l in range(1,no_of_layers):

        parameters['W'+str(l)]=np.random.randn(layers_dims_vector[l],layers_dims_vector[l-1])*0.01

        parameters['b'+str(l)]=np.zeros((layers_dims_vector[l],1))

    return parameters
def sigmoid(z):

    sigmoid = 1/(1 + np.exp(-z)) 

    return sigmoid,z



def relu(z):

    A = np.maximum(0,z)

    assert(A.shape == z.shape)

    return A,z
def linear_forward(W,b,A):

    Z = np.dot(W,A)+b

    cache = (A,W,b)

    return Z,cache

        
def linear_activation_forward(A_prev,W,b,activation):

    if(activation=="relu"):

        Z,linear_cache = linear_forward(W,b,A_prev)

        A,activation_cache = relu(Z)

    elif(activation=="sigmoid"):

        Z,linear_cache = linear_forward(W,b,A_prev)

        A,activation_cache = sigmoid(Z)

    cache = (linear_cache,activation_cache)

    return A,cache

    
def forward_prop(parameters,X):

    caches = []

    A = X

    L = len(parameters) // 2

    print(L)

    cache={}

    for l in range(1,L):

        A_prev = A

        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="relu")

        caches.append(cache)

        

     # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    ### START CODE HERE ### (≈ 2 lines of code)

    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="sigmoid")

    caches.append(cache)

    return AL,caches

            

    
def compute_cost(AL,Y):

    m=Y.shape[1]

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    assert(cost.shape == ())

    

    return cost
def linear_backward(dZ,cache):

    A_prev, W, b = cache

    m = A_prev.shape[1]



    ### START CODE HERE ### (≈ 3 lines of code)

    dW = 1/m*np.dot(dZ,A_prev.T)

    db = 1/m*np.sum(dZ,axis=1,keepdims=True)

    dA_prev = np.dot(W.T,dZ)

    ### END CODE HERE ###

    

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
def sigmoid_backward(dA,cache):

    Z = cache

    

    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)

    

    assert (dZ.shape == Z.shape)

    

    return dZ



def relu_backward(dA,cache):

    Z = cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    dZ[Z <= 0] = 0

    

    

    

    return dZ

    
def linear_activation_backward(dA,cache,activation):

    linear_cache,activation_cache = cache

    if(activation=="sigmoid"):

        dZ = sigmoid_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    elif(activation == "relu"):

        dZ = relu_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    

    return dA_prev,dW,db   
def back_prop(AL,Y,caches):

    grads = {}

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    L = len(caches)

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    ### END CODE HERE ###

    

    # Loop from l=L-2 to l=0

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        ### START CODE HERE ### (approx. 5 lines)

        current_cache = caches[l]

        #print(current_cache)

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

        ### END CODE HERE ###



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

    ### START CODE HERE ### (≈ 3 lines of code)

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]

        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    ### END CODE HERE ###

    return parameters
def l_layer_nn(no_of_iterations,learning_rate):

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions

    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T



    train_x = train_x_flatten/255

    test_x = test_x_flatten/255

    X = train_x

    Y= train_y

    no_of_units = [12288,20,7,5,1]



    parameters = initialize_parameters(no_of_units)

    #print(parameters)

    costs = []

    for i in range(0,no_of_iterations):

        AL,caches = forward_prop(parameters,X)

        #print("The forward Prop is:"+str(AL.shape))

        cost = compute_cost(AL,Y)

        grads = back_prop(AL,Y,caches)

        parameters = update_parameters(parameters,grads,learning_rate)

        print("Cost after iteration "+str(i)+" is:"+str(cost))

        costs.append(cost)

        

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

        

    

    
l_layer_nn(3000,0.0075)