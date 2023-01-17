# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#first importing the necessary librairies

import numpy as np



# created different function modules 



#y_predicted return predicted y-vector for given weights  and x



def y_predicted(w,x):

    y_pred = np.zeros(len(x))

    for i in range(0,len(x)):

        for j in range(0,len(w)):

             y_pred[i] = y_pred[i] + w[j]*x[i][j]

    return y_pred

#mse returns mean sqaure error 

def mse(y_actual,y_pred):

    error = 0

    for i in range(0,len(y_actual)):

        error = error + (y_actual[i] - y_pred[i])*(y_actual[i] - y_pred[i])

    return error/(2*len(y_actual))

#gradient calculates gradient vector

def gradient(y_actual,y_pred,x):

    grad = np.zeros(x.shape[1])

    for i in range(x.shape[1]):

        for j in range(0,len(y_actual)):

            grad[i] = - (y_actual[j] - y_pred[j])*x[j][i] + grad[i]

    return grad/len(y_actual)



#use lin_reg_weights to obtain linear weights. Specify no. of iterations and learning rate



def lin_reg_weights(x_train,y_train,num_iterations,learning_rate):

    row = x_train.shape[0]

    column = x_train.shape[1]

    new_x_train = np.ones((row,column+1))

    new_x_train[:,0:column] = x_train

    w = np.zeros(column+1)

    for i in range(0,num_iterations):

        y_pred = y_predicted(w,new_x_train)

        error = mse(y_train,y_pred)

        print("mean square error: ",error,"after",i,"th iteration")

        grad = gradient(y_train,y_pred,new_x_train)

        w = w - learning_rate*grad

    return w



# call linear_regression after obtaining weights in training data

def linear_regression(x_test,w):

    row = x_test.shape[0]

    column = x_test.shape[1]

    new_x_test = np.ones((row,column+1))

    new_x_test[:,0:column] = x_test

    y_pred = y_predicted(w,new_x_test)

    return(y_pred)

# call feature_scaling to normalize the values of variables

def feature_scaling(x):

    num_rows = x.shape[0]

    num_cols = x.shape[1]

    for i in range(num_cols):

        max_col  = np.amax(x[:,i])

        min_col = np.amin(x[:,i])

        for j in range(num_rows):

            x[j][i] = (x[j][i] - min_col)/(max_col-min_col) 

    return x



# Here is an example where I have used the above modules to predict housing prices for boston dataset

#testing the model using boston housing dataset



from sklearn.datasets import load_boston

boston = load_boston()

x = boston['data']

x = feature_scaling(x)

y = boston['target']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

w = lin_reg_weights(x_train,y_train,10,0.3)

y_pred = linear_regression(x_test,w)

print("mean square error: ",mse(y_test,y_pred))