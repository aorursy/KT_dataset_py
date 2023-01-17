# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()
data.info()
#specify x and y

y=data.Class.values

x_data=data.drop(['Class'],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#split data into train and test set

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.25, random_state=42)
x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T

print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initwb(dimension=30):

    w=np.full((dimension,1), 0.01)

    b=0.0

    return w, b
def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head
def fbprop(w, b, x_train, y_train):

    #forward propagation

    z=np.dot(w.T, x_train)+b

    y_head=sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    #sıfıra bölme(!!)

    cost=(np.sum(loss))/x_train.shape[1]

    #backward propagation

    dcost_dw=(np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]

    dcost_db=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={"derivative_weight":dcost_dw, "derivative_bias":dcost_db}

    return cost, gradients
def update(w, b, x_train, y_train, alpha, itera):

    cost_list=[]

    cost_list2=[]

    index=[]

    for i in range(itera):

        cost, gradients=fbprop(w, b, x_train, y_train)

        cost_list.append(cost)

        w=w-alpha*gradients["derivative_weight"]

        b=b-alpha*gradients["derivative_bias"]

        if i %100==0:

            cost_list2.append(cost)

            index.append(i)

            print("cost after iteration %i:%f" %(i,cost))

    parameters={"weight":w,"bias":b}

    plt.figure(figsize=(15,10))

    plt.plot(index,cost_list2)

    plt.show()

    return parameters,gradients,cost_list
def predict(w,b,x_test):

    z0=np.dot(w.T,x_test)+b

    z=sigmoid(z0)

    y_pred=np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            y_pred[0,i]=0

        else:

            y_pred[0,i]=1

    return y_pred
def logistic(x_train,y_train,x_test,y_test,alpha,itera):

    dimension = x_train.shape[0]

    w,b=initwb(dimension)

    parameters,gradients,cost_list=update(w, b, x_train, y_train, alpha, itera)

    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)    

    y_prediction_train=predict(parameters["weight"],parameters["bias"],x_train)

    print("train accuracy:{}%",format(100-np.mean(np.abs(y_prediction_train-y_train))*100))

    print("test accuracy:{}%",format(100-np.mean(np.abs(y_prediction_test-y_test))*100))





logistic(x_train,y_train,x_test,y_test,alpha=1,itera=500)