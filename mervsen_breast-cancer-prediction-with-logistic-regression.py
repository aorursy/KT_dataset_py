# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data.head()
data.info()
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values

x=data.iloc[:,1:]
#normalize x

x=(x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
def initial(dimension):

    w= np.full([dimension,1],0.01)

    b=0

    return w,b

def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head

    
def ForwardBackward(w,b,x_train,y_train):

    z= np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1]

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def Update(w,b,x_train,y_train,learning,itnumber):

    index=[]

    cost_list=[]

    for i in range(itnumber):

        cost,gradients=ForwardBackward(w,b,x_train,y_train)

        w=w-learning*gradients["derivative_weight"]

        b=b-learning*gradients["derivative_bias"]

        if(i%10==0):

            cost_list.append(cost)

            index.append(i)

            print("updated cost is {}".format(cost))

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, cost_list
def Predict(w,b,x_test):

    z=np.dot(w.T,x_test)+b

    z_=sigmoid(z)

    y_predict=np.zeros((1,x_test.shape[1]))

    for i in range(z_.shape[1]):

        if z[0,i]<=0.5:

            y_predict[0,i]=0

        else:

            y_predict[0,i]=1

    return y_predict





    
def LogReg(x_train,y_train,x_test,y_test,learning,itnumber):

    dim=x_train.shape[0]

    w,b = initial(dim)

    parameters,cost_list=Update(w,b,x_train,y_train,learning,itnumber)

    y_predict=Predict(parameters["weight"],parameters["bias"],x_test)

    print("Accuracy: {} %".format(100 - np.mean(np.abs(y_predict - y_test)) * 100))
import matplotlib.pyplot as plt

LogReg(x_train,y_train,x_test,y_test,learning=1,itnumber=30)
LogReg(x_train,y_train,x_test,y_test,learning=1.7,itnumber=300)