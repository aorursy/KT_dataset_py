# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import cv2 # For reading images

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir('../input/flowers/flowers'))
# Daisies path from Flower Recognation folder

daisy_path = "../input/flowers/flowers/daisy/"





#  Tulip path from Flower Recognation folder

tulip_path = "../input/flowers/flowers/tulip/" 
trainLabels = [] # For labels. Daisy and tulip

data = [] # All image array



# Dimensions of the images are not fixed. They have various sizes and we will fix tham to 128x128

size = 128,128



def readImages(flowerPath, folder):

    

    imagePaths = []

    for file in os.listdir(flowerPath):

        if file.endswith("jpg"):  # use only .jpg extensions

            imagePaths.append(flowerPath + file)

            trainLabels.append(folder)

            img = cv2.imread((flowerPath + file), 0)

            im = cv2.resize(img, size)

            data.append(im)            

            

    return imagePaths
def showImage(imgPath):

    img = cv2.imread(imgPath)

    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

    plt.axis('off')

    plt.show()
daisyPaths = readImages(daisy_path, 'daisy')

tulipPaths = readImages(tulip_path, 'tulip')
showImage(daisyPaths[np.random.randint(0,500)])

showImage(tulipPaths[np.random.randint(0,500)])
rawData = np.array(data)

rawData.shape
rawData = rawData.astype('float32') / 255.0
X = rawData

z = np.zeros(877)

o = np.ones(876)

Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0], 1)



print("X shape: " , X.shape)

print("Y shape: " , Y.shape)
# Let's create train and test data

from sklearn.model_selection import train_test_split



xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.15, random_state = 42)

numberOfTrain = xTrain.shape[0]

numberOfTest = xTest.shape[0]
# Transforming data to 2D.



xTrainFlatten = xTrain.reshape(numberOfTrain, xTrain.shape[1] * xTrain.shape[2])

xTestFlatten = xTest.reshape(numberOfTest, xTest.shape[1] * xTest.shape[2])



print("X train flatten", xTrainFlatten.shape)

print("X test flatten", xTestFlatten.shape)
x_train = xTrainFlatten.T

x_test = xTestFlatten.T

y_train = yTrain.T

y_test = yTest.T

print("x train: ",xTrain.shape)

print("x test: ",xTest.shape)

print("y train: ",yTrain.shape)

print("y test: ",yTest.shape)
def initializeParametersAndLayerSizesNN(x_train, y_train):

    

    parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3, 1)),

                  "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0], 1))}

    

    return parameters
# Method for sigmoid function

# z = np.dot(w.T, x_train) + b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
def forwardPropagationNN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
# Compute cost

def computeCostNN(A2, Y, parameters):

    

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    

    return cost
def backwardPropagationNN(parameters, cache, X, Y):



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
def updateParametersNN(parameters, grads, learning_rate):

    

    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],

                  "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],

                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],

                  "bias2": parameters["bias2"] - learning_rate * grads["dbias2"]}

    

    return parameters
# prediction

# x_test is the input of forward propagation.

def predictNN(parameters, x_test):



    A2, cache = forwardPropagationNN(x_test, parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(A2.shape[1]):

        if A2[0, i] <= 0.5:

            Y_prediction[0, i] = 0

        else:

            Y_prediction[0, i] = 1



    return Y_prediction
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    

    cost_list = []

    index_list = []

    

    # Initialize parameters

    parameters = initializeParametersAndLayerSizesNN(x_train, y_train)



    for i in range(0, num_iterations):

        # Forward propagation

        A2, cache = forwardPropagationNN(x_train, parameters)

        # Calculation of cost value

        cost = computeCostNN(A2, y_train, parameters)

         # Backward propagation

        grads = backwardPropagationNN(parameters, cache, x_train, y_train)

         # Updating parameters

        parameters = updateParametersNN(parameters, grads, learning_rate)

        

        if i % 10 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation = 'vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # Prediction

    y_prediction_test = predictNN(parameters, x_test)

    y_prediction_train = predictNN(parameters, x_train)



    # Print results

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 500)
# Reshaping for keras

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library



def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()



print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))