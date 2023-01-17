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
# load data set

X = np.load('../input/Sign-language-digits-dataset/X.npy')

y = np.load('../input/Sign-language-digits-dataset/Y.npy')
# Join a sequence of arrays along an row axis.

X = np.concatenate((X[204:409], X[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 

z = np.zeros(205)

o = np.ones(205)

y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)

print("y shape: " , y.shape)
#In order to use as input, we need to flatten the shape of X:

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

print(X.shape)



#410,4096 means that X data set have 410 samples and 4096 features
# Then lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = X_train.T

X_test = X_test.T

y_train = y_train.T

y_test = y_test.T



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# intialize parameters and layer sizes (we say layer size = 3)

def initialize_parameters_and_layer_sizes_NN(X_train, y_train):

    parameters = {"weight1": np.random.randn(3,X_train.shape[0]) * 0.1, #there are 3 nodes so w11,w12,w13 should be randomly created

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1, #there will be 1 output node, [(1x3)*(3x1)=1] - matrix multiplication rule

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters
def forward_propagation_NN(X_train, parameters):



    Z1 = np.dot(parameters["weight1"],X_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache

# Compute cost

def compute_cost_NN(A2, y, parameters): # prediction=A2, 

    #loss function:

    logprobs = np.multiply(np.log(A2),y)

    cost = -np.sum(logprobs)/y.shape[1] #sum of all losses. 

    return cost
# Backward Propagation

def backward_propagation_NN(parameters, cache, X, y):



    dZ2 = cache["A2"]-y #cost'un Z2'ye göre türevi

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1] #cost'un W2'ye göre türevi

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1] #cost'un B2'ye göre türevi

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2)) #cost'un Z1'ye göre türevi

    dW1 = np.dot(dZ1,X.T)/X.shape[1] #cost'un W1'ye göre türevi

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1] #cost'un B1'ye göre türevi

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2} #Bu dict, w1,w2,b1,b2'deki değişimleri depoladı.

    return grads
# update parameters

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# prediction

def predict_NN(parameters,X_test):

    # X_test is a input for forward propagation

    A2, cache = forward_propagation_NN(X_test,parameters)

    Y_prediction = np.zeros((1,X_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
#We need sigmoid function first.

# calculation of z

#z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



# 2 - Layer neural network

def two_layer_neural_network(X_train, y_train,X_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #initialize parameters and layer sizes

    parameters = initialize_parameters_and_layer_sizes_NN(X_train, y_train)



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation_NN(X_train,parameters)

        # compute cost

        cost = compute_cost_NN(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation_NN(parameters, cache, X_train, y_train)

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

    y_prediction_test = predict_NN(parameters,X_test)

    y_prediction_train = predict_NN(parameters,X_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(X_train, y_train,X_test,y_test, num_iterations=2500)
# load data set

X = np.load('../input/Sign-language-digits-dataset/X.npy')

y = np.load('../input/Sign-language-digits-dataset/Y.npy')
# Join a sequence of arrays along an row axis.

X = np.concatenate((X[204:409], X[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 

z = np.zeros(205)

o = np.ones(205)

y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)

print("y shape: " , y.shape)
#In order to use as input, we need to flatten the shape of X:

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

print(X.shape)



#410,4096 means that X data set have 410 samples and 4096 features
# Then lets create x_train, y_train, x_test, y_test arrays

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#importing functions:

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential #initialize neural network

from keras.layers import Dense #build our layers
#create method:

def build_classifier():

    classifier = Sequential() #initialize neural network

    classifier.add(Dense(units=8, kernel_initializer = 'uniform', activation='relu', input_dim=X_train.shape[1] ))

    classifier.add(Dense(units=4, kernel_initializer = 'uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier
ANN_classifier = KerasClassifier(build_fn= build_classifier, epochs=100)

accuracies = cross_val_score(estimator=ANN_classifier, X= X_train, y= y_train, cv=3)

mean = accuracies.mean()

variance = accuracies.std()

print('Accuracy mean:', mean)

print('Accuracy variance:', variance)