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
#reading data (read csv)

dataframe=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

print(dataframe.info())
#%% unnecessary  columns drop

dataframe=dataframe.drop(["Serial No."],axis=1) # or dataset.drop(["Unnamed: 32","id"],axis=1,inplace=True)
#%% x , y axis for model

y=dataframe.Research.values   # values  makes np array 

x_data=dataframe.drop(["Research"],axis=1) 
# normalization   feature scaling

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#%% train test splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#x_train(80%x), x_test(20%x) , y_train(80%y) , y_test(20%y) bu sıralama onu anlatıyor.

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
#sklearn with LR {test accuracy 0.73}



from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(x_train.T,y_train.T)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T))) 

# parameter initialize  and sigmoid function



# dimensions 30 

def initialize_weights_and_bias(dimensions):

    

    w=np.full((dimensions,1),0.01)

    b=0.0

    return w,b

   

def sigmoid(z):

    

    y_head=1/(1+np.exp(-z))

    return y_head

#forward and bacward propagation

 

# Forward propagation steps:

# find z = w.T*x+b

# y_head = sigmoid(z)

# loss(error) = loss(y,y_head)

# cost = sum(loss)

    

def forward_backward_propagation(w,b,x_train,y_train):

    #forward

    z=np.dot(w.T,x_train)+b # dot product array multiply and summation

    y_head=sigmoid(z)

    

    #loss function

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    #cost function

    cost=(np.sum(loss))/x_train.shape[1]# x_train.shape[1]  is for scaling

    

    #backward

    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

                                        # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

                                        # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}



    

    return cost,gradients
# Updating(learning) parameters



def update(w, b, x_train, y_train, learning_rate,number_of_iteration):

    cost_list = [] #for storage cost values

    cost_list2 = [] 

    index = []

    # updating(learning) parameters is number_of_iteration times

    for i in range(number_of_iteration): #

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    # plotting cost list regularly

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.01,number_of_iteration = 200)
#%% prediction method

 # prediction

def predict(w,b,x_test):# prediction yapmak için w,b ile oluşan modele x_test leri vereceğiz

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
#%% logistic regression total implement

    

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,number_of_iteration):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,number_of_iteration)



    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    

    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.15, number_of_iteration = 250)

#with this learning_rate = 0.15 and number_of_iteration = 500 : test accuracy is: 72.0 %)