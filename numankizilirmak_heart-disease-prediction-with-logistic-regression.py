# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataFrame=pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")

dataFrame.drop(['education'],axis=1,inplace=True)#remove irrevelant data

dataFrame.head()
dataFrame.info()
dataFrame.isnull().sum()# null values per column
dataFrame.dropna(axis=0,inplace=True) # drop null values 

dataFrame.isnull().sum()# now there ara no null values


sb.countplot(x='TenYearCHD',data=dataFrame) 
dataFrame.corr()
y=dataFrame.TenYearCHD.values

x=dataFrame.drop(["TenYearCHD"],axis=1)

x_norm=((x-np.min(x))/(np.max(x)-np.min(x))).values

x_train,x_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.2,random_state=42)# 20 % for test 80 % for train



x_train=x_train.T

y_train=y_train.T

x_test=x_test.T

y_test=y_test.T
#base methods

def init_weight_and_bias(dimension):



    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b



def sigmoid(z):

    y_head=1/(1 + np.exp(-z))

    return y_head
def forward_backward_propogation(w,b,x_train,y_train):

    #forward

    z=np.dot(w.T,x_train) + b # matrisleri çarp biası ekle (förmüldeki sum)

    y_head=sigmoid(z)

    loss=y_train * np.log(y_head) -(1-y_train)*np.log(1-y_head)

    cost=np.sum(loss)/x_train.shape[1] #x_train.shape[1] for scaling

    #backward

    deriative_weight=(np.dot(x_train,(y_head-y_train).T))/x_train.shape[1]

    deriative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={"derivative_weight":deriative_weight,"derivative_bias":deriative_bias}

    

    return cost,gradients



# Updating(learning) parameters

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propogation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    

    return parameters, gradients, cost_list





 # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1])) # sonuçlar için yeni matris oluştur

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)

    





def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = init_weight_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 250)

    

    