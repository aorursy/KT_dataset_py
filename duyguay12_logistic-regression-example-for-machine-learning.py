# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Read the file

data = pd.read_csv("../input/voice.csv")

print(data.info())
data.label = [1 if each == "female" else 0 for each in data.label]

print(data.info())
y = data.label.values

x_data = data.drop(["label"],axis=1)
"""

(x - min(x))/(max(x)-min(x))

"""

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
print(x_train.shape, y_train.shape)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print(x_train.shape, y_train.shape)
def initial_w_b(dimension):    

    w = np.full((dimension,1), 0.01) # (20,1)

    b = 0.0

    return w,b



def sigmoid(z):

    y_head = 1 / (1+np.exp(-z))

    return y_head



print(sigmoid(0)) # result must be 0.5 (test)
def propagation_forward_backward(w, b, x_train, y_train):

    # Forward propagation

    z = np.dot(w.T, x_train) + b

    y_head = sigmoid(z)

    loss = -(1-y_train)*np.log(1-y_head) - y_train*np.log(y_head) # formula, y:y_train

    cost = (np.sum(loss))/x_train.shape[1]  # x_train.shape[1] is for scaling

    

    # Backward Propagation

    derivative_weight = ( np.dot(x_train,((y_head-y_train).T)) )/x_train.shape[1] # x_train.shape[1] is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1] is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):

    cost_list = []

    cost_plot = []

    for i in range(number_of_iteration):

        # (1) make Forward and Backward propagation

        # (2) find the Cost and Gradients

        cost, gradients = propagation_forward_backward(w,b,x_train,y_train)

        cost_list.append(cost)   

        # Update:

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"] 

        

        # This section is just for plotting.

        if i % 10 == 0:

            cost_plot.append(cost)

            print ("Cost after iteration {0} : {1}".format(i, cost))

            

    # update (learn) the parameters; weights and bias

    parameters = {"weight": w,"bias": b}

    

    plt.plot(cost_plot)

    plt.xticks(rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters, gradients, cost_list
def prediction(w, b, x_test):

    # x_test is an input for the forward propagation

    y_head_test = sigmoid( np.dot(w.T , x_test) + b )

    Y_prediction = np.zeros( (1, x_test.shape[1]) )

    

    for i in range(y_head_test.shape[1]):

        if y_head_test[0,i] <= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

        

    # if y_head_test is bigger than 0.5, our prediction is one (Y_prediction=1),

    # if y_head_test is smaller than 0.5, our prediction is zero (Y_prediction=0)

    return Y_prediction
def logistic_regression (x_train, y_train, x_test, y_test, learning_rate ,  number_of_iteration):

    # initialize

    dimension = x_train.shape[0] # 20

    w, b = initial_w_b(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,number_of_iteration)

    

    y_prediction_test = prediction(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, number_of_iteration = 1000)    

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs')

# solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).

# Algorithm to use in the optimization problem.

# If we do not specify it, this error will be occured:

# FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning. FutureWarning)

lr.fit(x_train.T,y_train.T)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))