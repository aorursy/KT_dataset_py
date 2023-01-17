# libraries

import numpy as np                

import pandas as pd

import matplotlib.pyplot as plt
# read csv

data = pd.read_csv("../input/weatherAUS.csv")
# drop the unnecessary columns

data.drop(["Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm"], axis=1, inplace=True)
# convert "NO" to 0 and "YES" to 1

data.RainTomorrow = [ 1 if each == "Yes" else 0 for each in data.RainTomorrow ]

data.RainToday = [ 1 if each == "Yes" else 0 for each in data.RainToday ]



# convert Nan to 0

data = data.fillna(0)



data.head()
data.info()
# y is results. x is features without "RainTomorrow" and 'nan'

y = data.RainTomorrow.values

x = data.drop(["RainTomorrow"], axis=1)

# normalization

x = ((x - np.min(x))/(np.max(x) - np.min(x))).values
# train/test split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train shape: ", x_train.shape)

print("x_test shape: ", x_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)
# initialize weight and bias

def initialize_w_and_b(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
# sigmoid func

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# forward & backward propagation func

def forward_backward_propagation(w,b,x_train,y_train):

    # forward

    z = np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    

    # backward

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost, gradients
# updating

def update(w,b,x_train,y_train,learning_rate,num_iterations):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(num_iterations):

        cost,gradients = forward_backward_propagation(w,b,x_train, y_train)

        cost_list.append(cost)

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        param = { "weight": w, "bias": b }

        """if i % 50 == 0:

            index.append(i)

            cost_list2.append(cost)

            

    

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("iteation")

    plt.ylabel("cost")

    plt.show()"""  # wanna see graph, delete """

    

    return param, gradients, cost_list
# prediction

def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction
# logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    # initialize

    dimension = x_train.shape[0]   # that is 30

    w,b = initialize_w_and_b(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"], x_test)

    

    # print test error

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 300)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T, y_train.T)

print("test accuracy: {}".format(lr.score(x_test.T, y_test.T)))