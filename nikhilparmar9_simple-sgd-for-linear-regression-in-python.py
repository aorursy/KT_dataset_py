import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_boston

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from numpy import random

from sklearn.model_selection import train_test_split

print("DONE")
boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)

Y=load_boston().target

X=load_boston().data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
# data overview

boston_data.head(3)
print(X.shape)

print(Y.shape)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
## Before standardizing data

x_train
# standardizing data

scaler = preprocessing.StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test=scaler.transform(x_test)
## After standardizing data

x_train
x_test
## Adding the PRIZE Column in the data

train_data=pd.DataFrame(x_train)

train_data['price']=y_train

train_data.head(3)
x_test=np.array(x_test)

y_test=np.array(y_test)
type(x_test)
n_iter=100
# SkLearn SGD classifier

clf_ = SGDRegressor(max_iter=n_iter)

clf_.fit(x_train, y_train)

y_pred_sksgd=clf_.predict(x_test)

plt.scatter(y_test,y_pred_sksgd)

plt.grid()

plt.xlabel('Actual y')

plt.ylabel('Predicted y')

plt.title('Scatter plot from actual y and predicted y')

plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_sksgd))
# SkLearn SGD classifier predicted weight matrix

sklearn_w=clf_.coef_

sklearn_w
type(sklearn_w)
def My2CustomSGD(train_data,learning_rate,n_iter,k,divideby):

    w=np.zeros(shape=(1,train_data.shape[1]-1))

    b=0

    cur_iter=1

    while(cur_iter<=n_iter): 

#         print("LR: ",learning_rate)

        temp=train_data.sample(k)

        #print(temp.head(3))

        y=np.array(temp['price'])

        x=np.array(temp.drop('price',axis=1))

        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1))

        b_gradient=0

        for i in range(k):

            prediction=np.dot(w,x[i])+b

#             w_gradient=w_gradient+(-2/k)*x[i]*(y[i]-(prediction))

#             b_gradient=b_gradient+(-2/k)*(y[i]-(prediction))

            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))

            b_gradient=b_gradient+(-2)*(y[i]-(prediction))

        w=w-learning_rate*(w_gradient/k)

        b=b-learning_rate*(b_gradient/k)

        

        cur_iter=cur_iter+1

        learning_rate=learning_rate/divideby

    return w,b
def predict(x,w,b):

    y_pred=[]

    for i in range(len(x)):

        y=np.asscalar(np.dot(w,x[i])+b)

        y_pred.append(y)

    return np.array(y_pred)
w,b=My2CustomSGD(train_data,learning_rate=1,n_iter=100,divideby=2,k=10)

y_pred_customsgd=predict(x_test,w,b)



plt.scatter(y_test,y_pred_customsgd)

plt.grid()

plt.xlabel('Actual y')

plt.ylabel('Predicted y')

plt.title('Scatter plot from actual y and predicted y')

plt.show()

print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))
# weight vector obtained from impemented SGD Classifier

custom_w=w

print(custom_w)

print(type(custom_w))
w,b=My2CustomSGD(train_data,learning_rate=0.01,n_iter=1000,divideby=1,k=10)

y_pred_customsgd_improved=predict(x_test,w,b)



plt.scatter(y_test,y_pred_customsgd_improved)

plt.grid()

plt.xlabel('Actual y')

plt.ylabel('Predicted y')

plt.title('Scatter plot from actual y and predicted y')

plt.show()

print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd_improved))
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd_improved))
# weight vector obtained from impemented SGD Classifier

custom_w_improved=w

print(custom_w_improved)

print(type(custom_w_improved))
###

from prettytable import PrettyTable

x=PrettyTable()

x.field_names=['Model','Weight Vector','MSE']

x.add_row(['SKLearn SGD',sklearn_w,mean_squared_error(y_test, y_pred_sksgd)])

x.add_row(['Custom SGD',custom_w,mean_squared_error(y_test,y_pred_customsgd)])

x.add_row(['Custom SGD Improved',custom_w_improved,mean_squared_error(y_test,y_pred_customsgd_improved)])

print(x)