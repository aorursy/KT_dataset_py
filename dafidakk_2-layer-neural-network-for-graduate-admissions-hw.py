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
# DATA IMPORT

dataframe=pd.read_csv("../input/Admission_Predict.csv")



dataframe.info()
dataframe.head()
# DROP UNNECESSARY COLUMNS

dataframe=dataframe.drop(["Serial No.","Research"],axis=1)

dataframe.tail()



y1=dataframe["Chance of Admit "].values.reshape(400,1)



x1=(dataframe- np.min(dataframe))/(np.max(dataframe)-np.min(dataframe)).values



print("x1 shape: " , x1.shape)

print("y1 shape: " , y1.shape)

#normalization

x1=(dataframe- np.min(dataframe))/(np.max(dataframe)-np.min(dataframe)).values
x1.tail()
# TRAIN/TEST SPLIT SKLEARN

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.15, random_state=42)



#SHAPE TRAIN TEST 

x_test = x_test.T

x_train = x_train.T

y_test = y_test.T

y_train = y_train.T

print("X train: ",x_train.shape)

print("X test: ",x_test.shape)

print("Y train: ",y_train.shape)

print("Y test: ",y_test.shape)


# calculation of z with sigmoid fucntion between L2 layer to Output layer activation function

# z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# we have 1 hidden layer "L2" and input layer "L1" 

#it means we have two sets weights and biases to initialize to;

#intialize parameters and layer sizes

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters
# when doing forward propagation  we use tanh activation function to between L1 and L2

# after that at the output layer we use sigmoid 





def forward_propagation_NN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) + parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
# Cost function for Compute cost 

def compute_cost_NN(A2, y1, parameters):

    logprobs = np.multiply(np.log(A2),y1)

    cost = -np.sum(logprobs)/y1.shape[1]

    return cost

# Backward Propagation

# Gradient descent algorithm , this function helps update weights and biases with chained derivative calculus.

def backward_propagation_NN(parameters, cache, x1, y1):



    dZ2 = cache["A2"]-y1

    dW2 = np.dot(dZ2,cache["A1"].T)/x1.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/x1.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,x1.T)/x1.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/x1.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
# update parameters

# learning rate helps us how fast update parameters after backward propagation

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# we have output layer and this network gives an output, we use output as a prediction for comparison between  real data and output

def predict_NN(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.6,  prediction is Chance of Admit True  ,

    # if z is smaller than 0.6, prediction is Chance of Admit False,

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 2 - Layer neural network model

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

          # we plot cost every 100th calculation.

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





# Hiperparemeters are (learning_rate=0.01, number of iteration = 3000 and logistic treshold is 0.85 to Chance to Admission) 

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=3000)
#rehape for keras library



x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))