# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





from IPython.display import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.head()
data.columns
x_data = data.drop(["Outcome"],axis=1)

y = data.Outcome.values



# we seperate the result( Outcome column ) and other variables from each other.
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)



# we will do matrice product, for matrice product first matrix's column and second matrix's row must be same so we will transpose our train/test data



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
def fill_weights_and_bias(sizeofcolumn):

    w = np.full((sizeofcolumn,1),0.01)

    b = 0.00

    return w,b



# w = weights, b = bias 
def sigmoid(z):

    #sigmoid function returns y_head value

    y_head = (1 / ( 1 + np.exp(-z))) # its formula of sigmoid func.

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    # we must use weights and bias for training model

    # we must change w and b for appropriate shape to matrice product

    

    z = np.dot(w.T,x_train) + b

    

    y_head = sigmoid(z)

    

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #thats formula for our wrong predictions

    cost = (np.sum(loss))/x_train.shape[1] # thats average of loss 

    # forward propagation is completed

    

    # backward propagation

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    

    return cost,gradients



    #this func loop 1 times but we want to update our data as we learn new datas.
def update(w,b,x_train,y_train,learning_rate,loopnumber):

    

    cost_list = []

    cost_list2  =[]

    index = []

    

    # updating(learning) parameters is loopnumber times

    for i in range(1,loopnumber):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)

        cost_list.append(cost)

        

        # updating 

        w = w - learning_rate*gradients["derivative_weight"]

        b = b - learning_rate*gradients["derivative_bias"]

        

        # we may want information about progress

        if( i % 10 == 0):

            cost_list2.append(cost)

            index.append(i)

            print("Cost after {} times loop: {}".format(i,cost))

        

    # showing progress as visual is important

    parameters = {"weights" : w,"bias" : b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Loop")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

            

    # if z is bigger than 0.5, our prediction is sign one (y_head = 1)

    # if z is smaller than 0.5, our prediction is sign zero ( y_head = 0)

    

    for i in range(1,x_test.shape[1]):

        if (z[0,i] <= 0.5):

            y_prediction[0,i] == 0

        else:

            y_prediction[0,i] == 1

            

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,loopnumber):

    # initialize

    sizeofcolumn = x_train.shape[0]

    w,b = fill_weights_and_bias(sizeofcolumn)

    

    # forward and backward propagation

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,loopnumber)

    

    y_prediction_test = predict(parameters["weights"], parameters["bias"], x_test)

    # y_prediction_test our y values for test data now we will comparise each other

    

    #print test erros

    print("Test accuracy is: {}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100 ))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, loopnumber=600)