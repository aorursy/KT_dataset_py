# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.info()
df = data.drop(['age','sex'],axis = 1)
y = df.target.values

x_data = df.drop(['target'],axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) # normalization

x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T

print('x_train : ',x_train.shape)

print('x_test  : ',y_train.shape)

print('y_train : ',x_test.shape)

print('y_test  : ',y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("Accuary: % {}".format(lr.score(x_test.T,y_test.T)))
# 1 -  initialize_weights_and_bias

# 2 -  sigmoid

# 3 -  forward_and_backward

# 4 -  update

# 5 -  Predict

# 6 -  Logistic Regression
# 1

def initialize(demintion):

    w = np.full((demintion,1),0.01)

    b = 0.0

    return w,b
# 2

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# 3

def forward_and_backward(w,b,x_train,y_train):

    # forward

    z = np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)

    cost = np.sum((loss)) / x_train.shape[1]

    # backward

    derivative_weight = (np.dot(x_train,(y_head - y_train).T)) / x_train.shape[1]

    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]

    gradient = {'derivative_weight':derivative_weight,'derivative_bias':derivative_bias}

    return cost,gradient
# 4

def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(number_of_iteration):

        cost,gradient = forward_and_backward(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate * gradient['derivative_weight']

        b = b - learning_rate * gradient['derivative_bias']

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            if i == np.max(number_of_iteration):

                print('After iteration cost {} {}'.format(i,cost))

    parametres = {'weight':w,'bias':b}

    plt.subplots(figsize = (9,6))

    plt.plot(index,cost_list2)

    plt.grid()

    plt.xlabel('Number of Iteration',fontsize = 15)

    plt.ylabel('Cost',fontsize = 15)

    plt.plot()

    return cost_list,parametres
# 5 

def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_predict = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_predict[0,i] = 0

        else:

            y_predict[0,i] = 1

    return y_predict
# 6

def logistic(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):

    demintion = x_train.shape[0]

    w,b = initialize(demintion)

    cost_list,parametres= update(w,b,x_train,y_train,learning_rate,number_of_iteration)

    y_predict_test = predict(parametres['weight'],parametres['bias'],x_test)

    print(' Test Accuary: % {}'.format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))
logistic(x_train,y_train,x_test,y_test,learning_rate = 1,number_of_iteration = 300)