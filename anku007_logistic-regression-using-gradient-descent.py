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
import numpy as np

import math



def sigmoid(z):

    return 1/(1+math.exp(-z))



def y_predicted(w,x):

    y_pred = np.zeros(len(x))

    for i in range(0,len(x)):

        for j in range(0,len(w)):

             y_pred[i] = y_pred[i] + w[j]*x[i][j]

        y_pred[i] = sigmoid(y_pred[i])

    return y_pred



def cross_entropy(y_actual,y_pred):

    error = 0

    for i in range(len(y_actual)):

        error = -(y_actual[i])*math.log(y_pred[i])-(1-y_actual[i])*math.log(1-y_pred[i]) + error

    return error/len(y_actual)



def classification(y):

    for i in range(len(y)):

        if y[i]>=0.5:

            y[i] = 1

        else:

            y[i] = 0

    return y



def classification_score(y_actual,y_pred):

    true_as_true = 0

    true_as_false = 0

    false_as_false = 0

    false_as_true = 0

    for i in range(len(y_actual)):

        if y_actual[i] ==1 and y_pred[i]==1:

            true_as_true  = true_as_true + 1

        elif y_actual[i] ==1 and y_pred[i]==0:

            true_as_false  = true_as_false + 1

        elif y_actual[i] ==0 and y_pred[i]==0:

            false_as_false = false_as_false + 1

        elif y_actual[i] ==0 and y_pred[i]==1:

            false_as_true = false_as_true + 1

    return (true_as_true + false_as_false)/len(y_actual)



def gradient(y_actual,y_pred,x):

    grad = np.zeros(x.shape[1])

    for i in range(x.shape[1]):

        for j in range(0,len(y_actual)):

            grad[i] = - (y_actual[j] - y_pred[j])*x[j][i] + grad[i]

    return grad/len(y_actual)



def reg_weights(x_train,y_train,num_iterations,learning_rate):

    row = x_train.shape[0]

    column = x_train.shape[1]

    new_x_train = np.ones((row,column+1))

    new_x_train[:,0:column] = x_train

    w = np.zeros(column+1)

    for i in range(0,num_iterations):

        y_pred = y_predicted(w,new_x_train)

        error = cross_entropy(y_train,y_pred)

        #print("cross_entropy",error,"after",i,"th iteration")

        y_pred_class=classification(y_pred)

        score = classification_score(y_train,y_pred_class)

        #print("classification score",score,"after",i,"th iteration")

        grad = gradient(y_train,y_pred,new_x_train)

        w = w - learning_rate*grad

    return w



def logistic_regression(x_test,w):

    row = x_test.shape[0]

    column = x_test.shape[1]

    new_x_test = np.ones((row,column+1))

    new_x_test[:,0:column] = x_test

    y_pred = y_predicted(w,new_x_test)

    return y_pred



def feature_scaling(x):

    num_rows = x.shape[0]

    num_cols = x.shape[1]

    for i in range(num_cols):

        max_col  = np.amax(x[:,i])

        min_col = np.amin(x[:,i])

        for j in range(num_rows):

            x[j][i] = (x[j][i] - min_col)/(max_col-min_col) 

    return x

# Applying it on cancer datatset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data['data']

y = data['target']



x = feature_scaling(x)



from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)



w = reg_weights(x_train,y_train,100,1)



y_pred = logistic_regression(x_test,w)

error = cross_entropy(y_test,y_pred)

print("cross_entropy:",error)

y_pred_class = classification(y_pred)

score = classification_score(y_test,y_pred_class)

print("classification score: ",score)
