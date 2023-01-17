# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/diabetes.csv")
data.head()
data.info()
data.isnull().values.any()
data.describe()
data.columns = map(str.lower, data.columns)

data.columns
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.distplot(data.age, bins = 20, ax=ax[0,0]) 

sns.distplot(data.pregnancies, bins = 20, ax=ax[0,1]) 

sns.distplot(data.glucose, bins = 20, ax=ax[1,0]) 

sns.distplot(data.bloodpressure, bins = 20, ax=ax[1,1]) 

sns.distplot(data.skinthickness, bins = 20, ax=ax[2,0])

sns.distplot(data.insulin, bins = 20, ax=ax[2,1])

sns.distplot(data.diabetespedigreefunction, bins = 20, ax=ax[3,0]) 

sns.distplot(data.bmi, bins = 20, ax=ax[3,1]) 
sns.regplot(x = data.pregnancies, y = data.glucose)
sns.set(font_scale = 1.15)

plt.figure(figsize = (14, 10))



sns.heatmap(data.corr(), vmax = 1, linewidths = 0.5, fmt= '.1f',

            square = True, annot = True, cmap = 'YlGnBu', linecolor = "white")

plt.title('Correlation of Features');
# Normalization

# Normalization Formula; (x - min(x))/max(x)-min(x)

y = data.outcome.values

x = data.drop(["outcome"], axis = 1)



x = (x - np.min(x))/(np.max(x)-np.min(x)).values
# Train & Test Split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



features = x_train.T

labels = y_train.T

test_features = x_test.T

test_labels = y_test.T



print("features: ", features.shape)

print("labels: ", labels.shape)

print("test_features: ", test_features.shape)

print("test_labels: ", test_labels.shape)



#Parameter Initialize 

def initialize_weights_and_bias(dimension):

    w = np.full((dimension, 1),0.01)

    b= 0.0

    return w,b
# Sigmoid Function**

# Sigmoid Function Formula; 1/(1+e^-x)

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head

# Forward & Backward Propagation

def foward_and_backward_propagation(w, b, x_train, y_train):

    #Forward Propagation

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]          #x_train.shape[1] is for scaling

    

    # Backward Propagation

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost, gradients
#Updating Parameters

def update(w, b, x_train, y_train, learning_rate, number_of_iterations):

    cost_list = []

    cost_list2 = []

    index = []

    

    # Updating (learning) parameters is number_of_iterations times

    for i in range(number_of_iterations):

        

        cost, gradients = foward_and_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost)

        #Let's update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iterations %i: %f" %(i, cost))

            

    # We update (learn) parameters weights and bias

    parameters = {"weight": w, "bias": b}

    plt.plot(index, cost_list2)

    plt.title("Cost-Iteration Relation")

    plt.xticks(index, rotation = "vertical")

    plt.xlabel("Number of iterations")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
#Prediction

def predict(w, b, x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T, x_test)+b)

    y_prediction = np.zeros((1, x_test.shape[1]))

    #

    #

    for i in range(z.shape[1]):

        if z[0, i] <= 0.5:

            y_prediction[0, i] = 0

        else:

            y_prediction[0, i] = 1

            

    return y_prediction
# Logistic Regression

def logistic_regression(features, labels, test_features, test_labels, learning_rate ,  num_iterations):

    # Initialize

    dimension =  features.shape[0]  # It is 8

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, features, labels, learning_rate,num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],test_features)

    # Print test errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - test_labels)) * 100))

    

logistic_regression(features, labels, test_features, test_labels,learning_rate = 1.5, num_iterations = 300)   
# Logistic Regression with Scikit-Learn

from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(features.T, labels.T).score(test_features.T, test_labels.T)))
labels = labels.reshape(labels.shape[0], -1).T

test_labels = test_labels.reshape(test_labels.shape[0], -1).T



print(labels.shape)

print(test_labels.shape)
class Artificial_Neural_Network(object):

    

    def __init__(self, xTrain, xTest, yTrain, yTest):

        # Define train and test data

        self.xTrain = xTrain

        self.xTest = xTest

        self.yTrain = yTrain.reshape(yTrain.shape[0],-1)

        self.yTest = yTest.reshape(yTest.shape[0],-1)



        # Define hyperparameters

        self.inputLayerSize = self.xTrain.shape[0] # nx <-> Number of samples

        self.hiddenLayerSize = 4

        self.outputLayerSize = self.yTrain.shape[0]

        

    def initializeWeightsAndBias(self): #, inputLayerSize, hiddenLayerSize, outputLayerSize):

        """

        This function creates a vector of zeros of shape (inputLayerSize, 1) for w and initializes b to 0.



        Argument:

        inputLayerSize -- size of the input layer

        hiddenLayerSize -- size of the hidden layer

        outputLayerSize -- size of the output layer



        Returns:

        params -- python dictionary containing your parameters:

                        W1 -- weight matrix of shape (hiddenLayerSize, inputLayerSize)

                        b1 -- bias vector of shape (hiddenLayerSize, 1)

                        W2 -- weight matrix of shape (outputLayerSize, hiddenLayerSize)

                        b2 -- bias vector of shape (outputLayerSize, 1)

        """

        np.random.seed(23) # We set up a seed so that your output matches ours 

                           # although the initialization is random.

        

        W1 = np.random.randn(self.inputLayerSize, 

                             self.hiddenLayerSize) * 0.01

        b1 = np.zeros(shape=(self.hiddenLayerSize, 1))

        W2 = np.random.randn(self.hiddenLayerSize,

                             self.outputLayerSize) * 0.01

        b2 = np.zeros(shape=(self.outputLayerSize, 1))

        

        # assert(isinstance(B1, float) or isinstance(B1, int))

        

        assert (W1.shape == (self.inputLayerSize, self.hiddenLayerSize)), "[W1] -> Unsuitable matrix size"

        assert (b1.shape == (self.hiddenLayerSize, 1))

        assert (W2.shape == (self.hiddenLayerSize, self.outputLayerSize)), "[W2] -> Unsuitable matrix size"

        assert (b2.shape == (self.outputLayerSize, 1))

        

        parameters = {"W1": W1,

                      "b1": b1,

                      "W2": W2,

                      "b2": b2}   

        

        return parameters

    

    def sigmoid(self, Z):

        """ Apply and compute sigmoid activation function to scalar, vector, or matrix (Z)



        Arguments:

        Z -- A scalar or numpy array of any size.



        Return:

        s -- sigmoid(z)

        """

        return 1/(1+np.exp(-Z))

    

    def forwardPropagation(self, X, parameters):

        """ Propogate inputs though network

        

        Argument:

        X -- input data of size (inputLayerSize, m)

        parameters -- python dictionary containing your parameters (output of initialization function)



        Returns:

        A2 -- The sigmoid output of the second activation

        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

        """

        # Retrieve each parameter from the dictionary "parameters"

        W1 = parameters['W1']

        b1 = parameters['b1']

        W2 = parameters['W2']

        b2 = parameters['b2']



        # Implement Forward Propagation to calculate A2 (probabilities)

        Z1 = np.dot(W1.T, X) + b1

        A1 = sigmoid(Z1)

        Z2 = np.dot(W2.T, A1) + b2

        yHat = self.sigmoid(Z2) # A2

        

        assert(yHat.shape == (1, X.shape[1]))

    

        cache = {"Z1": Z1,

                 "A1": A1,

                 "Z2": Z2,

                 "yHat": yHat}    # A2

    

        return yHat, cache

    

    def computeCost(self, yHat, Y, parameters):

        """ Compute cost for given X,Y, use weights already stored in class 



        Arguments:

        yHat -- The sigmoid output of the second activation, of shape (1, number of examples)

        Y -- "true" labels vector of shape (1, number of examples)

        parameters -- python dictionary containing your parameters W1, b1, W2 and b2



        Returns:

        cost -- cross-entropy cost given equation (13)

        """

        m = Y.shape[1] # number of example

                      

        # Retrieve W1 and W2 from parameters

        W1 = parameters['W1']

        W2 = parameters['W2']   

                    

        # Loss

        logprobs = np.multiply(np.log(yHat), Y) + np.multiply((1 - Y), np.log(1 - yHat))

        # Cost

        cost = - (np.sum(logprobs)) / m     # m =  yTrain.shape[1]  is for scaling

        

        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

                                    # E.g., turns [[17]] into 17 

        assert(isinstance(cost, float))

                      

        return cost



    def backwardPropagation(self,parameters, cache,  X, Y):

        """ Compute the gradients of parameters by implementing the backward propagation



        Arguments:

        parameters -- python dictionary containing our parameters 

        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".

        X -- input data of shape (2, number of examples)

        Y -- "true" labels vector of shape (1, number of examples)



        Returns:

        grads -- python dictionary containing your gradients with respect to different parameters

        """

        m = X.shape[1]   

                      

        # First, retrieve W1 and W2 from the dictionary "parameters".       

        W1 = parameters['W1']

        W2 = parameters['W2']

                      

        # Retrieve also A1 and A2 from dictionary "cache".

        A1 = cache['A1']

        yHat = cache['yHat']                    

                      

        # Backward propagation: calculate dW1, db1, dW2, db2.                     

        dZ2 = yHat - Y

        dW2 = (1 / m) * np.dot(A1, dZ2.T)

        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.multiply(np.dot(W2, dZ2), 1 - np.power(A1, 2))

        dW1 = (1 / m) * np.dot(X, dZ1.T)#(1 / m) * np.dot(dZ1, self.xTrain.T) # MATRIS BOYUTLARINA BAK dW1 ve dW2 ICIN

        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)   # m is for scaling 



        gradients = {"dW1": dW1,

                     "db1": db1,

                     "dW2": dW2,

                     "db2": db2}

                      

        return gradients

    

    def updateParameters(self, parameters, gradients, learning_rate = 0.15):

        """

        Updates parameters using the gradient descent update rule given above



        Arguments:

        parameters -- python dictionary containing your parameters 

        grads -- python dictionary containing your gradients 



        Returns:

        parameters -- python dictionary containing your updated parameters 

        """

        # Retrieve each parameter from the dictionary "parameters"

        W1 = parameters['W1']

        b1 = parameters['b1']

        W2 = parameters['W2']

        b2 = parameters['b2']



        # Retrieve each gradient from the dictionary "grads"

        dW1 = gradients['dW1']

        db1 = gradients['db1']

        dW2 = gradients['dW2']

        db2 = gradients['db2']

        

        # Update rule for each parameter

        W1 = W1 - learning_rate * dW1

        b1 = b1 - learning_rate * db1

        W2 = W2 - learning_rate * dW2

        b2 = b2 - learning_rate * db2



        parameters = {"W1": W1,

                      "b1": b1,

                      "W2": W2,

                      "b2": b2}



        return parameters

                      

    def model(self, X, Y, num_iterations=10000, print_cost=False):

        """

        Arguments:

        X -- dataset of shape (2, number of examples)

        Y -- labels of shape (1, number of examples)

        n_h -- size of the hidden layer

        num_iterations -- Number of iterations in gradient descent loop

        print_cost -- if True, print the cost every 1000 iterations



        Returns:

        parameters -- parameters learnt by the model. They can then be used to predict.

        """

        np.random.seed(3)

        

        costStr = []

        indexStr = []

        

        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".

        parameters = self.initializeWeightsAndBias()



        W1 = parameters['W1']

        b1 = parameters['b1']

        W2 = parameters['W2']

        b2 = parameters['b2']

 

        # Loop (gradient descent)

        for i in range(0, num_iterations):

                      

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".

            yHat, cache = self.forwardPropagation(X, parameters)



            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".

            cost = self.computeCost(yHat, Y, parameters)

            

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".

            gradients = self.backwardPropagation(parameters, cache, X, Y)



            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".

            parameters = self.updateParameters(parameters, gradients, learning_rate = 0.0001)



            # Print the cost every 1000 iterations

            if print_cost and i % 1000 == 0:

                costStr.append(cost)

                indexStr.append(i)

                print ("Cost after iteration %i: %f" % (i, cost))

            

        # Plot Cost Function

        plt.plot(indexStr,costStr)

        plt.xticks(indexStr,rotation='vertical')

        plt.xlabel("Number of Iterarion")

        plt.ylabel("Cost")

        plt.show()

            

        return parameters



    def predict(self, parameters, X):

        """

        Using the learned parameters, predicts a class for each example in X



        Arguments:

        parameters -- python dictionary containing your parameters 

        X -- input data of size (n_x, m)



        Returns

        predictions -- vector of predictions of our model (red: 0 / blue: 1)

        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

        yHat, cache = self.forwardPropagation(X, parameters)

        predictions = np.round(yHat)



        

        return predictions
ANN = Artificial_Neural_Network(features, test_features, labels, test_labels)

parameters = ANN.model(features, labels, num_iterations = 25000, print_cost=True)

predictions = ANN.predict(parameters, features)

print('Train Accuracy: %d' % float((np.dot(labels, predictions.T) + np.dot(1 - labels, 1 - predictions.T)) / float(labels.size) * 100) + '%')