import numpy as np

import math

import pandas

import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split
# load dataset

dataframe = pandas.read_csv("../input/ecoli.csv", delim_whitespace=True)

# Assign names to Columns

dataframe.columns = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'site']

dataframe = dataframe.drop('seq_name', axis=1)

# Encode Data

dataframe.site.replace(('cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'),(1,0,0,0,0,0,0,0), inplace=True)

dataset = dataframe.values

#print (dataset[0])

#print dataset shape

DataX = np.array(dataset[:,0:7])

print (DataX.shape)

DataY = np.transpose([dataset[:,7]])

print (DataY.shape)
X_train,  X_test,Y_train, Y_test = train_test_split(DataX, DataY, test_size = 0.2)

print (X_train.shape, Y_train.shape)

print (X_test.shape, Y_test.shape)
#defining the neural network structure

def layer_sizes(X, Y):

    n_x = X.shape[0] # size of input layer

    n_h = 5

    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
(n_x, n_h, n_y) = layer_sizes(X_train, Y_train)

print("The size of the input layer is: n_x = " + str(n_x))

print("The size of the hidden layer is: n_h = " + str(n_h))

print("The size of the output layer is: n_y = " + str(n_y))
# gradient function： initialize parameter

def initialize_parameters( n_h, X):

    np.random.seed(3)

    m, n = X.shape

    W1 = np.random.randn(n_h,n) * 0.001

    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(1, n_h) * 0.001

    b2 = np.zeros((m, 1))

    parameters = {"W1" : W1,

                  "b1" : b1,

                  "W2" : W2, 

                  "b2" : b2}

    return parameters

    
#initialize_parameters = initialize_parameters( 5, X_train)

#print (initialize_parameters["W1"].shape) (5, 7)

#print (initialize_parameters["W2"].shape) (1, 5)
#gradent function: forward_propagation

def forward_propagation(X, parameters, Y):

    W1 = parameters["W1"]

    #b1 = parameters["b1"]

    

    W2 = parameters["W2"]

   # b2 = parameters["b2"]

    

    Z1 = np.dot(W1, X.T)

    yhat1 = np.tanh(Z1)

    

    Z2 = np.dot(W2, yhat1).T

    yhat2 =result = 1 / (1 + np.exp(- Z2))

    #compute cost

    m = Y.size

    logprobs = np.multiply(Y, np.log(yhat2)) + np.multiply((1 - Y), np.log(1 - yhat2))

    cost = - 1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)

    cache = { "Z1" : Z1,

              "yhat1" : yhat1,

              "Z2" : Z2,

              "yhat2" : yhat2}

    yhat2 = cache["yhat2"]

    return yhat2, cache

    print ("The cost is : %s" % cost)

    print (W2.shape,yhat1.shape,np.dot(W2, yhat1).shape,b2.shape,Z2.shape,yhat2.shape)



#parameters = initialize_parameters(n_h, X_train)

#forward_propagation(X_train, parameters, Y_train)

#cache = forward_propagation(X_train, parameters, Y_train)
def compute_cost(yhat2, Y, parameters):

    m = Y.size

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    logprobs = np.multiply(np.log(yhat2), Y) + np.multiply((1 - Y), np.log(1 - yhat2))

    cost = - 1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)

    return cost
#gradient function : backward_propagation

def backward_propagation(parameters, cache, X, Y):

     m = X.shape[1]

    

     # First, retrieve W1 and W2 from the dictionary "parameters".

     ### START CODE HERE ### (≈ 2 lines of code)

     W1 = parameters["W1"]

     W2 = parameters["W2"]

     ### END CODE HERE ###

        

     # Retrieve also yhat1 and yhat2 from dictionary "cache".

     ### START CODE HERE ### (≈ 2 lines of code)

     yhat1 = cache["yhat1"]

     yhat2 = cache["yhat2"]

     ### END CODE HERE ###

     

     #db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

     #Backward_propagation: caculate dw1, dw2, d

     ##start code here##

     dZ2= yhat2 - Y

     dW2 = (1 / m) * np.dot(dZ2.T,yhat1.T)

     #db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

     dZ1 = np.multiply(np.dot(W2.T, dZ2.T), (1 - np.power(yhat1, 2)))

     dW1 = (1 / m) * np.dot(dZ1, X)

     ##end code here##

     grads = {"dZ2" : dZ2,

              "dW2" : dW2,

              "dZ1" : dZ1,

              "dW1" : dW1}

     return grads     
#grads = backward_propagation(parameters, cache, X_train, Y_train)

#print ("The dZ2 shape is " + str(grads["dZ2"].shape))

#print ("The dW2 shape is " +  str(grads["dW2"].shape))

#print ("The dZ1 shape is " +  str(grads["dZ1"].shape))

#print ("The dW1 shape is " +  str(grads["dW1"].shape))
def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    
def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    

    dW1 = grads["dW1"]

    dW2 = grads["dW2"]

    

    W1 -= dW1 * learning_rate

    W2 -= dW2 * learning_rate

    

    update_parameters = {"W1" : W1,

                         "W2" : W2}

    return update_parameters
#update_parameters = update_parameters(parameters, grads, 0.01)

#print ("update_parameters W1 is: " + str(update_parameters["W1"].shape))

#print ("update_parameters W2 is: " + str(update_parameters["W2"].shape))
def SNN_Model(X_train, Y_train, n_h, learning_rate, iater_num ):

    

    ###we are random initialize the train parameters###

    ### start code here ###

    np.random.seed(1)

    parameters = initialize_parameters(n_h, X_train)

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    ###end code here ###

    

    # Loop (gradient descent)

    for i in range(0, iater_num):

        

    ### We use forward paropagation  to caculate linear part and activate part of tanh or sigmod 

        yhat2, cache = forward_propagation(X_train, parameters, Y_train)

    

    ###cost function

        cost = compute_cost(yhat2, Y_train, parameters)

        

    ### We use the backward propagation to caculate dW1 and dW2

        grads = backward_propagation(parameters, cache, X_train, Y_train)

    

    ### We use learning rate to caculate the patameters after update

        updateparameters = update_parameters(parameters, grads, learning_rate)

        #print (updateparameters)

        #print (parameters["W1"].shape)

        #print (updateparameters["W1"].shape)

        parameters["W1"] = updateparameters["W1"]

        parameters["W2"] = updateparameters["W2"]

       # if print_cost and i % 1000 == 0:

           # print ("cost after iteration %i : %f" % (i, cost))

    #return update_parameters["W1" : W1, "W2" : W2]

    return parameters

    
###Using the train data to test the previous function

SNN_Model = SNN_Model(X_train, Y_train, 5, 0.01, 100)

print (SNN_Model["W1"].shape)

print (SNN_Model["W2"].shape)
def predict(parameters, X_test, Y_test):

    

    yhat2, cache = forward_propagation(X_test, parameters, Y_test)

    #predictions = (yhat2 > 0.5) ###vector of predictions values is True or False

    

    return yhat2#predictions
#SNN_Model = SNN_Model(X_train, Y_train, 5, 0.01, 100)

predictions = predict(SNN_Model, X_test, Y_test)

print(predictions.shape)

print("Y_test mean = " + str(np.mean(Y_test)))

print("predictions mean = " + str(np.mean(predictions)))
###Calculating the accuracy

predictions = predict(SNN_Model, X_test, Y_test)

print ("The accuracy is: " + str(100 * np.mean((np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size))))