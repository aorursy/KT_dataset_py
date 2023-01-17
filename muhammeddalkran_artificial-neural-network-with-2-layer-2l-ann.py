import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load data set

x_l = np.load('../input/sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(x_l[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis('off')

# Then lets create x_train, y_train, x_test, y_test arrays

# Join a sequence of arrays along an row axis.

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)

print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def parameter_initialize(x_train,y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0])*0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3)*0.1,

                  "bias2" : np.zeros((y_train.shape[0],1))}

    

    print("Shape of weight1 : ",parameters["weight1"].shape)

    print("Shape of weight2 : ",parameters["weight2"].shape)

    print("Shape of bias1 : ",parameters["bias1"].shape)

    print("Shape of bias2 : ",parameters["bias2"].shape)



    return parameters

parameter_initialize(x_train,y_train)
def sigmoid(z):

    A = 1/(1+np.exp(-z))

    return A
def forward_propagation(x_train,parameters):

    Z1 = np.dot(parameters["weight1"],x_train) + parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)

    

    results = {"Z1": Z1, "A1":A1,"Z2":Z2,"A2":A2}

    return A2, results
def cost(A2,Y):

    logaritmic_probability = np.multiply(np.log(A2),Y)

    cost = -np.sum(logaritmic_probability)/Y.shape[1]

    return cost
def backward_propagation(parameters,results,X,Y):

    dZ2 = results["A2"]-Y

    dW2 = np.dot(dZ2,results["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2, axis = 1, keepdims = True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1-np.power(results["A1"],2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1, axis=1,keepdims = True)/X.shape[1]

    gradients = {"dweight1": dW1,

                "dweight2": dW2,

                "dbias1": db1,

                "dbias2":db2}

    return gradients
def update_prameters(parameters,grand,learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grand["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grand["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grand["dweight2"],

                  "bias2" : parameters["bias2"]-learning_rate*grand["dbias2"]

                 }

    return parameters
def prediction(parameters, x_test):

    A2, results = forward_propagation(x_test,parameters)

    prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(A2.shape[1]):

        if A2[0,i] <= 0.5:

            prediction[0,i] = 0

        else:

            prediction[0,i] = 1

    return prediction
def two_layer_ANN_model(x_train, y_train, x_test, y_test, number_of_iteration):

    cost_list = []

    index = []

    parameters = parameter_initialize(x_train,y_train)

    for i in range(number_of_iteration):

        A2, results = forward_propagation(x_train,parameters)

        cost_result = cost(A2,y_train)

        gradients = backward_propagation(parameters,results,x_train,y_train)

        parameters = update_prameters(parameters, gradients)

        

        if i % 100 == 0:

            cost_list.append(cost_result)

            index.append(i)

            print("Cost after iteration %i %f" %(i,cost_result))

    plt.plot(index,cost_list)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of ıteration")

    plt.ylabel("Cost")

    plt.show()

    y_prediction_test = prediction(parameters,x_test)

    y_prediction_train = prediction(parameters,x_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters

parameters = two_layer_ANN_model(x_train,y_train,x_test,y_test,2000)