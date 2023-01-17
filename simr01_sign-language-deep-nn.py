# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

data = np.load('../input/Sign-language-digits-dataset/X.npy')

labels = np.load('../input/Sign-language-digits-dataset/Y.npy')

data.shape, labels.shape # data have 2062 observations, each consists in a 64x64 px

                         # in labels, we have same numbers of row in 10 col 



def plot_lang_digits(data, label):

    print('Dataset examples:')

    number_of_classes = label.shape[1]

    col_idx = [i for i in range(number_of_classes)]

    plt.figure(figsize=(18, 6))



    for i in col_idx:

        ax = plt.subplot(2, 5, i+1) # row col index

        ax.set_title("idx: " + str(i))

        plt.imshow(data[np.argwhere(label[:,i]==1)[0][0],:]) # , cmap='gray'

        #plt.gray()

        plt.axis('off')

        

plot_lang_digits(data, labels)
'index is not correct'

# we create dictionary and match the corresponding

# label_idx_map = { 0:9, 1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}

label_idx_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}



Y = np.zeros(data.shape[0]) # create array of '0' with data size as given data (2062, 64, 64)

#  fill array with correct labels

Y[:204] = 9; Y[204:409] = 0; Y[409:615] = 7; Y[615:822] = 6; Y[822:1028] = 1; Y[1028:1236] = 8; Y[1236:1443] = 4; 

Y[1443:1649] = 3; Y[1649:1855] = 2; Y[1855:] = 5

'Analysing only 0s & 1s'

X = np.concatenate((data[204:409], data[822:1027]), axis=0) # Images of all Zeros and Ones 

# creating labels for 0 and 1 sign images.

z=np.zeros(205)

o=np.ones(205)

Y=np.concatenate((z,o), axis=0).reshape(X.shape[0],1)

# The shape of X is (410, 64, 64), where:

# 410 images (zero and one signs), 64 means that image size is 64x64 (64x64 pixels)

# The shape of the Y is (410,1), where:

# 410 means that we have 410 labels (0 and 1)  , X.shape[0] = 410



print(" X shape: ",  X.shape, "\n Y shape: ",  Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

print(' X_train.shape :',X_train.shape,'\n X_test.shape  :',X_test.shape)

# 3 Dimensional input array X

# Y (labels) is already in 2D

# X (image array) to be Flattened from 3D array to 2D

X_train_flatten = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

X_test_flatten = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

print(' X_train_flatten :',X_train_flatten.shape ,'\n X_test_flatten  :',X_test_flatten.shape)

# We have 348 and 62 image arrays and each image has 4096 pixels

# Transpose each of the four image arrays

x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print(" x_train:",x_train.shape, 

      "\n x_test: ",x_test.shape,

      "\n y_train:",y_train.shape,

      "\n y_test: ",y_test.shape)

'Initilizing parameters'

#np.full((3, 1), 7, dtype=int) 3 is height



def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01) # initialize weight 0.01

    b = 0.0 # initialize bias 0

    return w, b





w,b = initialize_weights_and_bias(4096)
'Forward Propagation'



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



# y_head = sigmoid(0)

# y_head



# find z = w.T*x+b

# calculation of z is: z = np.dot(w.T,x_train)+b

# y_head = sigmoid(z) # probabilistic 0-1

# loss(error) = loss(y,y_head)

# cost = sum(loss)



def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) 

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 
'Optimization Algorithm with Gradient Descent'

#w,b = initialize_weights_and_bias(1)

cost = forward_propagation(w,b,x_train,y_train)

cost # without any iteration or 0 iteration
# In backward propagation we will use y_head that found in forward progation

# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

   

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    

    return cost,gradients
# Updating(learning) parameters

def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009, number_of_iterarion = 200)
 # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
#Applying all functions

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate , num_iterations):

    # initialize

    dimension = x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
'Logistic Regression with Sklearn'

from sklearn import linear_model



logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)



print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# intialize parameters and layer sizes

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters



def forward_propagation_NN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache



# Compute cost

def compute_cost_NN(A2, Y, parameters):

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost



# Backward Propagation

def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads



# update parameters

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters



# prediction

def predict_NN(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #initialize parameters and layer sizes

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation_NN(x_train,parameters)

        # compute cost

        cost = compute_cost_NN(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

         # update parameters

        parameters = update_parameters_NN(parameters, grads)

        

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # predict

    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)