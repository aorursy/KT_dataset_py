# import necessary libraries

import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_boston

from random import seed

from random import randrange

from csv import reader

from math import sqrt

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error,mean_absolute_error

from numpy import random

from sklearn.model_selection import train_test_split
boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)

Y=load_boston().target

X=load_boston().data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
# data overview

boston_data.head(3)
# standardizing data

scaler = preprocessing.StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test=scaler.transform(x_test)
train_data=pd.DataFrame(x_train)

train_data['price']=y_train

train_data.head(3)
x_test=np.array(x_test)

y_test=np.array(y_test)
# shape of test and train data matxis

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# SkLearn SGD classifier

clf_ = SGDRegressor()

clf_.fit(x_train, y_train)

plt.scatter(y_test,clf_.predict(x_test))

plt.grid()

plt.xlabel('Actual y')

plt.ylabel('Predicted y')

plt.title('scatter plot between actual y and predicted y')

plt.show()

print('Mean Squared Error :',mean_squared_error(y_test, clf_.predict(x_test)))

print('Mean Absolute Error :',mean_absolute_error(y_test, clf_.predict(x_test)))
# SkLearn SGD classifier predicted weight matrix

sklearn_w=clf_.coef_

sklearn_w
# implemented SGD Classifier

def CustomGradientDescentRegressor(train_data,learning_rate=0.001,n_itr=1000,k=10):

    w_cur=np.zeros(shape=(1,train_data.shape[1]-1))

    b_cur=0

    cur_itr=1

    while(cur_itr<=n_itr):

        w_old=w_cur

        b_old=b_cur

        w_temp=np.zeros(shape=(1,train_data.shape[1]-1))

        b_temp=0

        temp=train_data.sample(k)

        #print(temp.head(3))

        y=np.array(temp['price'])

        x=np.array(temp.drop('price',axis=1))

        for i in range(k):

            w_temp+=x[i]*(y[i]-(np.dot(w_old,x[i])+b_old))*(-2/k)

            b_temp+=(y[i]-(np.dot(w_old,x[i])+b_old))*(-2/k)

        w_cur=w_old-learning_rate*w_temp

        b_cur=b_old-learning_rate*b_temp

        if(w_old==w_cur).all():

            break

        cur_itr+=1

    return w_cur,b_cur

def predict(x,w,b):

    y_pred=[]

    for i in range(len(x)):

        y=np.asscalar(np.dot(w,x[i])+b)

        y_pred.append(y)

    return np.array(y_pred)





def plot_(test_data,y_pred):

    #scatter plot

    plt.scatter(test_data,y_pred)

    plt.grid()

    plt.title('scatter plot between actual y and predicted y')

    plt.xlabel('actual y')

    plt.ylabel('predicted y')

    plt.show()        

        
# Funtion to get optimal learning rate on the implemented SGD Classifier

from math import log

x1_train,x1_test,y1_train,y1_test=train_test_split(X,Y,test_size=0.3)

x1_train,x1_cv,y1_train_,y1_cv_=train_test_split(x1_train,y1_train,test_size=0.3)



x1_train = scaler.transform(x1_train)

x1_cv=scaler.transform(x1_cv)



x1_train_=np.array(x1_train)

x1_train_data=pd.DataFrame(x1_train)

x1_train_data['price']=y1_train_



x1_cv_data=pd.DataFrame(x1_cv)

x1_cv_data['price']=y1_cv_



y1_train_=np.array(y1_train_)

y1_cv_=np.array(y1_cv_)

#print(y1_cv_.shape)



def tuneParams_learning_rate():

    train_error=[]

    cv_error=[]

    r=[0.00001,0.0001,0.001,0.01,0.1]

    for itr in r:

        w,b=CustomGradientDescentRegressor(x1_train_data,learning_rate=itr,n_itr=1000)

       # print(w.shape,b.shape,x1_train_.shape)

        y1_pred_train=predict(x1_train_,w,b)

        train_error.append(mean_squared_error(y1_train_,y1_pred_train))

        w,b=CustomGradientDescentRegressor(x1_cv_data,learning_rate=itr,n_itr=1000)

        y1_pred_cv=predict(x1_cv,w,b)

        cv_error.append(mean_squared_error(y1_cv_,y1_pred_cv))

    return train_error,cv_error 



    

        
train_error,cv_error=tuneParams_learning_rate()
  # plotting obtained values

import math

r=[0.00001,0.0001,0.001,0.01,0.1]

x1=[math.log10(i) for i in r]

plt.plot(x1,train_error,label='train MSE')

plt.plot(x1,cv_error,label='CV MSE')

plt.scatter(x1,train_error)

plt.scatter(x1,cv_error)

plt.legend()

plt.xlabel('log of learning rate')

plt.ylabel('Mean Squared Error')

plt.title('log(learning rate) vs MSE')

plt.grid()

plt.show()
# running implemented SGD Classifier with obtained optimal learning rate

w,b=CustomGradientDescentRegressor(train_data,learning_rate=0.001,n_itr=1000)

y_pred=predict(x_test,w,b)

plot_(y_test,y_pred)
# Errors in implemeted model

print(mean_squared_error(y_test,y_pred))

print(mean_absolute_error(y_test,y_pred))
# weight vector obtained from impemented SGD Classifier

custom_w=w

custom_w
from prettytable import PrettyTable

# MSE = mean squared error

# MAE =mean absolute error

x=PrettyTable()

x.field_names=['Model','Weight Vector','MSE','MAE']

x.add_row(['sklearn',sklearn_w,mean_squared_error(y_test, clf_.predict(x_test)),mean_absolute_error(y_test, clf_.predict(x_test))])

x.add_row(['custom',custom_w,mean_squared_error(y_test,y_pred),(mean_absolute_error(y_test,y_pred))])

print(x)
sklearn_pred=clf_.predict(x_test)

implemented_pred=y_pred

x=PrettyTable()

x.field_names=['SKLearn SGD predicted value','Implemented SGD predicted value']

for itr in range(15):

    x.add_row([sklearn_pred[itr],implemented_pred[itr]])

print(x)   