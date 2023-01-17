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
data = pd.read_csv("../input/candy-data.csv")
data.info()

# I do not need "competitorname" so I drop it

data.drop("competitorname", inplace = True, axis=1)
# initialize x and y

y = data.chocolate.values

x_data = data.drop(["chocolate"], axis = 1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train shape", x_train.shape)

print("x_test shape", x_test.shape)

print("y_train shape", y_train.shape)

print("y_test shape", y_test.shape)
def initialize_weights_and_bias(dimension):

    """

    Input

    dimension => number of train_data's features

    

    Output

    w => weights

    b => bias

    """

    w = np.full((dimension,1), 0.01)

    b = 0

    return w, b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head

    
def forward_backward_propagation(w, b, x_train, y_train):

    """

    Input

    w => weights

    b => bias

    x_train => x of the data we want the train

    y_train => y of the data we want the train

    

    Output

    cost => loss of function

    gradients => derivative of weights and bias

    """

    

    # forward propagation

    z = (np.dot(w.T, x_train)+b)

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-((1-y_train)*np.log(1-y_head))

    cost = np.sum(loss)/x_train.shape[1]

    

    # backward propagation

    

    derivative_weights = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weights":derivative_weights, "derivative_bias":derivative_bias}

    

    return cost, gradients

    
def update(w, b, x_train, y_train, learning_rate, num_iteration):

    """

    Input

    w => weights

    b => bias

    x_train => x of the data we want the train

    y_train => y of the data we want the train

    learning_rate => learn speed (if speed is too big, function can not be run properly)

    num_iteration => how many times we want to run forward_backward_propagation()

    

    Output

    paremeter => last values of weights and bias

    cost_list => all cost(loss) values we got

    """

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(num_iteration):

        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost)

        

        w = w - learning_rate*gradients["derivative_weights"]

        b = b - learning_rate*gradients["derivative_bias"]

    

        

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration {}: {}".format(i, cost))

    

    parameters = {"weights":w, "bias":b}        

    plt.plot(index, cost_list2)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters, gradients, cost_list

            
def predict(w, b, x_test):

    """

    Input

    w => last values of weights

    b => last value of bias

    x_test => x of the data we want to test

    

    Output

    y_predict => prediction of the test data

    """

    z = sigmoid(np.dot(w.T, x_test)+b)

    y_predict = np.zeros((1, x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_predict[0,i] = 0

        else:

            y_predict[0,i] = 1

    return y_predict
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iteration):

    w, b = initialize_weights_and_bias(x_train.shape[0])

    

    parameter, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iteration)

    

    y_predict = predict(parameter["weights"], parameter["bias"], x_test)

    print("accuracy: {}".format(100 - np.mean(np.abs(y_predict - y_test))))

    

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iteration = 100)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



lr.fit(x_train.T, y_train.T)



print("Test accuracy: {}".format(lr.score(x_test.T, y_test.T)))

print("Train accuracy: {}".format(lr.score(x_train.T, y_train.T)))