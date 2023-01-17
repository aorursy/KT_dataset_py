# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weatherAUS.csv')

data.sample(5)
# Getting rid of the columns with objects which will not be used in our model:

data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM'], axis=1, inplace=True)

data.head(5)
# And we need to replace NaN values with mean values of each column:

data.fillna(data.mean(), inplace=True)

data.head(5)
# Now we can change that day and next days'predictions (yes and no) to 1 and 0:

data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]

data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]

data.sample(3)
y = data.RainTomorrow.values

x_data = data.drop('RainTomorrow', axis=1)

x_data.head()
# In order to scale all the features between 0 and 1:

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head(5)
# importing sklearn's library for splitting our dataset:

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)



# For our matrix calculations we need to transpose our matrixis:

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T



print('x_train shape is: ', x_train.shape)

print('y_train shape is: ', y_train.shape)

print('x_test shape is: ', x_test.shape)

print('y_test shape is: ', y_test.shape)
def initialize_weight_bias(dimension):

    w = np.full((dimension,1), 0.01)    # Create a matrix by the size of (dimension,1) and fill it with the values of 0.01

    b = 0.0

    return w,b
def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head
def forward_backward_propagation(w, b, x_train, y_train):

    # forward propagation:

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    

    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)     # loss function formula

    cost = (np.sum(loss)) / x_train.shape[1]                               # cost function formula

    

    # backward propagation:

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    

    gradients = {'derivative_weight': derivative_weight, 'derivative_bias': derivative_bias}

    

    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, nu_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    

    # Initialize for-back propagation for the number of iteration times. Then updating w and b values and writing the cost values to a list:  

    for i in range(nu_of_iteration):

        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost)

    

        # Update weight and bias values:

        w = w - learning_rate * gradients['derivative_weight']

        b = b - learning_rate * gradients['derivative_bias']

        # Show every 20th value of cost:

        if i % 20 == 0:

            cost_list2.append(cost)

            index.append(i)

            print('Cost after iteration %i: %f' %(i,cost))

    

    parameters = {'weight': w, 'bias':b}

    

    # Visulization of cost values:

    plt.plot(index, cost_list2)

    plt.xlabel('Nu of Iteration')

    plt.ylabel('Cost Function Value')

    plt.show()

    

    return parameters, gradients, cost_list
def prediction(w, b, x_test):

    z = sigmoid(np.dot(w.T, x_test) + b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

            

    return y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, nu_of_iteration):

    dimension = x_train.shape[0]

    w, b = initialize_weight_bias(dimension)    # Creating an initial weight matrix of (x_train data[0] x 1)

    

    # Updating our w and b by using update method. 

    # Update method contains our forward and backward propagation.

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, nu_of_iteration)

    

    # Lets use x_test for predicting y:

    y_test_predictions = prediction(parameters['weight'], parameters['bias'], x_test) 

    

    # Investigate the accuracy:

    print('Test accuracy: {}%'.format(100 - np.mean(np.abs(y_test_predictions - y_test))*100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, nu_of_iteration=400)
# Importing sklearn library for logistic regression:

from sklearn.linear_model import LogisticRegression



# Creating our model named 'lr'

lr = LogisticRegression()



# Training it by using our train data:

lr.fit(x_train.T, y_train.T)



# Printing our accuracy by using our trained model and test data:

print('Test accuracy of sklearn logistic regression library: {}'.format(lr.score(x_test.T, y_test.T)))