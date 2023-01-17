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
#read data set

df = pd.read_csv("../input/train_and_test2.csv")

df.info()

df.head()
df.drop(["zero","zero.1","zero.2","zero.3","zero.4","zero.5","zero.6","zero.7","zero.8","zero.9","zero.10","zero.11","zero.12","zero.13","zero.14","zero.15","zero.16","zero.17","zero.18"],axis=1,inplace = True)
df.drop(["Passengerid"],axis=1,inplace = True)


df = df.rename(columns={"2urvived":"Survived"})

df.tail()
df = df.astype(float)

df.Embarked.value_counts( dropna=False)

df.dropna(axis=0,inplace=True)
x_data = df.drop(["Survived"],axis=1)

y = df.Survived.values
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

x_train =x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train",x_train.shape,"x_test:",x_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape)
x_train.shape[0]
def initialize_weight_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
#sigmoid function

# z = np.dot (w.T,x)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# definition of foward and backward procedures

def foward_and_backward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = np.sum(loss)/x_train.shape[1]

    #backward

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients
def update(w,b,x_train,y_train,learning_rate,number_iterations):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_iterations):

        cost,gradients = foward_and_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w-learning_rate*gradients["derivative_weight"]

        b = b-learning_rate*gradients["derivative_bias"]

        

        if i%10 ==0:

            cost_list2.append(cost)

            index.append(i)

            print("cost after iteration %i:%f"%(i,cost))

            

    parameters = {"weights": w, "bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation = "vertical")

    plt.xlabel("iteration")

    plt.ylabel("cost")        

    plt.show()

    return parameters,cost_list,gradients

    

    
def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test))

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            Y_prediction[0,i] =0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate, number_iterations):

    

    dimension = x_train.shape[0]

    w,b = initialize_weight_and_bias(dimension)

    

    parameters,gradients, cost_list = update(w,b,x_train,y_train,learning_rate,number_iterations)

    

    y_prediction_test = predict(parameters["weights"],parameters["bias"],x_test)

    

    print("test accuracy:{} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))



logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, number_iterations = 1000)