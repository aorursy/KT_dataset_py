import numpy as np                                 #importing the standard library for scientific calculation

import seaborn as sns                              #importing interative plotting library

import matplotlib.pyplot as plt                    #importing a plotting library

import pandas as pd                                #importing this library to read and store files

from sklearn.model_selection import train_test_split  #this is used to split the data into training and test set

from sklearn.metrics import classification_report, confusion_matrix   #These are some of the testing metrics we would be using in the course of our program
def sigmoid(Z):                                   #defining the sigmoid function that we would use in the later parts of the code

    """

    Implements the sigmoid activation in numpy

    

    Arguments:

    Z -- numpy array of any shape

    

    Returns:

    A -- output of sigmoid(z), same shape as Z

    cache -- returns Z as well, useful during backpropagation

    """

    

    A = 1/(1+np.exp(-Z))                          #This is the definition of sigmoid function

    cache = Z                                     #Storing some values in cache which could be used in back propagation later

    

    return A, cache                               #Returning the cache and the activation A



def relu(Z):                                      #Defining relu function that is used in various layers

    """

    Implement the RELU function.



    Arguments:

    Z -- Output of the linear layer, of any shape



    Returns:

    A -- Post-activation parameter, of the same shape as Z

    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently

    """

    

    A = np.maximum(0,Z)                         #Definition of relu

    

    assert(A.shape == Z.shape)                  #Ensuring that the shape stays consistent

    

    cache = Z                                   #storing the value Z in cache which would later be used in deep neural networks

    return A, cache                             #returning the value of cache and activation





def relu_backward(dA, cache):                   #This is used for taking the derivatives of relu while using back propagation

    """

    Implement the backward propagation for a single RELU unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache                                  #Storing the values in cache

    dZ = np.array(dA, copy=True) # just converting dz to a correct object. 

    

    # When z <= 0, you should set dz to 0 as well. 

    dZ[Z <= 0] = 0

    

    assert (dZ.shape == Z.shape)

    

    return dZ



def sigmoid_backward(dA, cache):      #This is used for sigmoid backward function and in backpropagation

    """

    Implement the backward propagation for a single SIGMOID unit.



    Arguments:

    dA -- post-activation gradient, of any shape

    cache -- 'Z' where we store for computing backward propagation efficiently



    Returns:

    dZ -- Gradient of the cost with respect to Z

    """

    

    Z = cache 

    

    s = 1/(1+np.exp(-Z))

    dZ = dA * s * (1-s)                 #This is the derivative which would later be returned

    

    assert (dZ.shape == Z.shape)        #Ensuring that the shape stays consistent

    

    return dZ



df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df.columns
df['University Rating'].value_counts()
df['SOP'].value_counts()
sns.color_palette("Paired")

sns.distplot(df['SOP'], color = 'brown')
df['LOR '].value_counts()
sns.distplot(df['LOR '], color = 'darkorange')
df.tail()
sns.barplot(x = 'Research', y = 'GRE Score', data = df)
sns.jointplot(x = 'TOEFL Score', y = 'GRE Score', color = 'darkblue', data = df)
sns.jointplot(x = 'TOEFL Score', y = 'GRE Score', kind = 'kde', data = df, color = 'pink')
sns.jointplot(x = 'CGPA', y = 'GRE Score', kind = 'hex', data = df)
sns.jointplot(x = 'TOEFL Score', y = 'GRE Score', kind = 'reg', data = df, color = 'g')
sns.jointplot(x = 'SOP', y = 'LOR ', kind = 'kde', color = 'g', data = df)
sns.jointplot(x = 'SOP', y = 'LOR ', kind = 'hex', color = 'r', data = df)
df.columns
X = df.drop(['Chance of Admit ', 'Serial No.'], axis = 1) #here we would drop the columns that are not necessary for the input such as the serial number and chance of admit

y = df['Chance of Admit ']                                #we would just require the ouput to be Chance of Admit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)  #we are dividing the input along with the output into training and testing set

X_train = X_train.T                                  #We take the transpose of the training set for simplicity

X_test = X_test.T                                    #We take the transpose of the testing set for simplicity

y_train = y_train[:, np.newaxis]                     #We want to convert it into a n-dimensional vector

y_test = y_test[:, np.newaxis]                       #We convert it to n-dimensional vector

y_train = y_train.T                                  #We take the transpose for simplicity

y_test = y_test.T                                    #We take the transpose for simplicity

y = y[:, np.newaxis]

X = X.T                                              #We take the transpose for simplicity

y = y.T                                              #We take the transpose for simplicity

print('the shape of the input is {}'.format(X.shape))    #here we would print the shape of the input

print('the shape of the output is {}'.format(y.shape))   #printing the shape of the output

print('the shape of the input training set is {}'.format(X_train.shape))  #printing the shape of input training set

print('the shape of the output training set is {}'.format(y_train.shape)) #printing the shape of output training set

print('the shape of the input training set is {}'.format(X_test.shape))   #printing the shape of input testing set

print('the shape of the output training set is {}'.format(y_test.shape))  #printing the shape of output testing set
def initialize_parameters(n_x, n_h, n_y):     #this is a function used to initialize the weights and biases

    w1 = np.random.randn(n_h, n_x) * 0.01     #we use xavier initialization in this process 

    b1 = np.zeros((n_h, 1))                   #we create an array of zeroes

    w2 = np.random.randn(n_y, n_h) * 0.01     #we use xavier initialization in this process

    b2 = np.zeros((n_y, 1))                   #we create an array of zeroes

    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}  #we load these parameters into a dictionary so that they can be used later

    return parameters                         #we return the parameters to the function 
def initialize_parameters_deep(layer_dims):   #This function is used to create weights and biases for all the L layers in the neural network

    L = len(layer_dims)                       #We take the dimensions of the input in the function

    parameters = {}                           #We create an empty dictionary so that it could be accessed later in the function

    for l in range(1, L):                     #We create a for loop to iterate through all the layers

        parameters['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01  #We use xavier initialization for all the L-layers

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))                              #We create zeroes for the L-layer network

    return parameters                         #We return the parameters
def linear_forward(a, w, b):  #We propagate forward through the network in this function

    z = np.dot(w, a) + b      #We take the dot product between the weight and the activation and add a bias to it

    cache = (a, w, b)         #We store all these values in cache so that they could be accessed later

    return z, cache           #We return these parameters to the function
def linear_activation_forward(a_prev, w, b, activation): #We create this function to calculate both the linear part and also the activation parts in the network

    if activation == "sigmoid":                          #If the activation is sigmiod, we process the information using sigmoid

        z, linear_cache = linear_forward(a_prev, w, b)   #We save the ouput in z and linear_cache

        a, activation_cache = sigmoid(z)                 #We then apply the sigmoid to z to produce activation a

    elif activation == "relu":                           #If the activation is relu, we process the information using relu

        z, linear_cache = linear_forward(a_prev, w, b)   #We save the ouput in z and linear_cache

        a, activation_cache = relu(z)                    #We then apply the relu to z to produce activation a

    cache = (linear_cache, activation_cache)             #We save both the linear cache and activation cache in cache that could later be used

    return a, cache                                      #We return both the activation and the cache
def L_model_forward(X, parameters):                     #We create a model for the forward propagation of the network. 

    caches = []                                         #We define a new list to store the values 

    a = X                                               #The given value of X is taken as a

    L = len(parameters) // 2                          

    for l in range(1, L):

        a_prev = a

        a, cache = linear_activation_forward(a_prev, parameters['w' + str(l)], parameters['b' + str(l)], "relu")

        caches.append(cache)

    al, cache = linear_activation_forward(a, parameters['w' + str(L)], parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return al, caches
def compute_cost(a, y):

    m = y.shape[1]

    cost = -(1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    cost = np.squeeze(cost)

    return cost
def linear_backward(dz, cache):

    a_prev, w, b = cache

    m = a_prev.shape[1]

    dw = (1 / m) * np.dot(dz, a_prev.T)

    db = (1 / m) * np.sum(dz, axis = 1, keepdims = True)

    da_prev = np.dot(w.T, dz)

    return da_prev, dw, db
def linear_activation_backward(da, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":

        dz = relu_backward(da, activation_cache)

        da_prev, dw, db = linear_backward(dz, linear_cache)

    elif activation == "sigmoid":

        dz = sigmoid_backward(da, activation_cache)

        da_prev, dw, db = linear_backward(dz, linear_cache)

    return da_prev, dw, db

        
def L_model_backward(al, y, caches):

    grads = {}

    L = len(caches)

    m = al.shape[1]

    y = y.reshape(al.shape)

    dal = - (np.divide(y, al) - np.divide(1 - y, 1 - al))

    current_cache = caches[L - 1]

    grads["da" + str(L-1)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dal, current_cache, activation = "sigmoid")

    for l in reversed(range(L - 1)):

        current_cache = caches[l]

        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads["da" + str(l + 1)], current_cache, activation = "relu")

        grads["da" + str(l)] = da_prev_temp

        grads["dw" + str(l + 1)] = dw_temp

        grads["db" + str(l + 1)] = db_temp

    return grads

    

    

    
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):

        parameters["w" + str(l + 1)] = parameters["w" + str(l + 1)] - learning_rate * grads["dw" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters     
def two_layer_model(X, y, layers_dims, learning_rate = 0.001, num_iterations = 10000, print_cost = False):

    grads = {}

    costs = []

    m = X.shape[1]

    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    w1 = parameters["w1"]

    b1 = parameters["b1"]

    w2 = parameters["w2"]

    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        a1, cache1 = linear_activation_forward(X, w1, b1, activation = "relu")

        a2, cache2 = linear_activation_forward(a1, w2, b2, activation = "sigmoid")

        cost = compute_cost(a2, y)

        da2 = - (np.divide(y, a2) - np.divide(1 - y, 1 - a2))

        da1, dw2, db2 = linear_activation_backward(da2, cache2, activation = "sigmoid")

        da0, dw1, db1 = linear_activation_backward(da1, cache1, activation = "relu")

        grads["dw1"] = dw1

        grads["db1"] = db1

        grads["dw2"] = dw2

        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        w1 = parameters["w1"]

        b1 = parameters["b1"]

        w2 = parameters["w2"]

        b2 = parameters["b2"]

        if print_cost and i % 1000 == 0:

            print("cost after iteration {}:: {}".format(i, np.squeeze(cost)))

        if print_cost and i % 1000 == 0:

            costs.append(cost)

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    return parameters    
layers_dims = [X.shape[0], 50, 25, 1]
def L_layer_model(X, y, layers_dims, learning_rate = 0.03, num_iterations = 3000, print_cost = False):

    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        al, caches = L_model_forward(X, parameters)

        cost = compute_cost(al, y)

        grads = L_model_backward(al, y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:

            print("Cost after iteration {}: {}".format(i, cost))

        if print_cost and i % 1000 == 0:

            costs.append(cost)

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate = " + str(learning_rate))

    plt.show()

    return parameters
parameters = L_layer_model(X_train, y_train, layers_dims,learning_rate = 0.05, num_iterations = 60000, print_cost = True)
def predict(parameters, X):

    a2, cache = L_model_forward(X, parameters)

    predictions = a2

    return predictions
predictions_test = predict(parameters, X_test)

predictions_train = predict(parameters, X_train)

print('The accuracy of the training model: {}%'.format((1 - np.sum(np.abs(predictions_train - y_train))/predictions_train.shape[1]) * 100))

print('The accuracy of the testing model: {}%'.format((1 - np.sum(np.abs(predictions_test - y_test))/predictions_test.shape[1]) * 100))