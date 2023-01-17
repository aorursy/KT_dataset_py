import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/logistic-regression/Social_Network_Ads.csv")



df.drop(["User ID","Gender"],axis=1,inplace=True)



y = df.Purchased.values

x_data = df.drop(["Purchased"],axis=1)



x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.reshape(-1,1).T

y_test = y_test.reshape(-1,1).T
def init_weights_and_bias(dimension):

    weights = np.full((dimension,1),0.01)

    bias = 0

    return weights,bias



weights,bias = init_weights_and_bias(2)



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
def f_b_propagation(weights,bias,x_train,y_train):

    z = np.dot(weights.T,x_train) + bias

    

    y_head = sigmoid(z)

    

    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1-y_head)

    

    cost = (np.sum(loss)) / x_train.shape[1]

    

    der_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]

    der_bias = np.sum(y_head - y_train) / x_train.shape[1]   

    

    gradients = {"der_weight": der_weight,"der_bias": der_bias}

    

    return cost,gradients
def update(weights,bias,x_train,y_train,learning_rate,number_of_iteration):

    cost_list = []

    cost_list_2 = []

    index = []

    

    for i in range(number_of_iteration):

        cost,gradients = f_b_propagation(weights,bias,x_train,y_train)

        cost_list.append(cost)

        

        weights = weights - learning_rate * gradients["der_weight"]

        bias = bias - learning_rate * gradients["der_bias"]

        

        if i % 10 == 0:

            cost_list_2.append(cost)

            index.append(i)

            print(f"Cost after iteration {i}:{cost}")

            

    parameters = {"weights": weights,"bias": bias}

    plt.plot(index,cost_list_2)

    

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(weights,bias,x_test):

    z = sigmoid(np.dot(weights.T, x_test) + bias)

    Y_prediction = np.zeros((1, x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def log_regr(x_train,y_train,x_test,y_test,learning_rate,iter_num):

    dimension = x_train.shape[0]

    weights,bias = init_weights_and_bias(dimension)

    

    parameters, gradients, cost_list = update(weights, bias, x_train, y_train, learning_rate, iter_num)

    

    y_prediction_test = predict(parameters["weights"],parameters["bias"],x_test)

    

log_regr(x_train, y_train, x_test, y_test, learning_rate = 0.01, iter_num = 150)
from sklearn.linear_model import LogisticRegression



logr = LogisticRegression()



logr.fit(x_train.T,y_train.T)



print("Test accuracy: {}".format(logr.score(x_test.T,y_test.T)))