# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/voicegender/voice.csv")
data.columns
data = data.rename({'label': 'gender'}, axis=1)

data.gender = [1 if each == "female" else 0 for each in data.gender]
y = data.gender.values

x1 = data.drop(["gender"],axis=1)
x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1)).values
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=24)

x_train = x_train.T

x_test = x_test.T
def weight_bias(dia):

    w = np.full((dia,1),0.01)

    b = 0.0

    return w,b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



def forwardbackward(w,b,x_train,y_train):

    # forward

    z = np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    # backward

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}

    

    return cost,gradients
def update(w, b, x_train, y_train, learningrate,numberofiterarion):

    costlist = []

    costlist2 = []

    index = []



    for i in range(numberofiterarion):



        cost,gradients = forwardbackward(w,b,x_train,y_train)

        costlist.append(cost)

     

        w = w - learningrate * gradients["derivative_weight"]

        b = b - learningrate * gradients["derivative_bias"]

        if i % 10 == 0:

            costlist2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))



    parameters = {"weight": w,"bias": b}

    plt.plot(index,costlist2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, costlist
def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5,prediction is 0 else 1 for y_head

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logisticregression(x_train, y_train, x_test, y_test, learningrate ,  numiterations):

    

    dia=  x_train.shape[0]

    w,b = weight_bias(dia)

    

    parameters, gradients, costlist = update(w, b, x_train, y_train, learningrate,numiterations)

    

    yprediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    print("test accuracy: {} %".format(100 - np.mean(np.abs(yprediction_test - y_test)) * 100))
logisticregression(x_train, y_train, x_test, y_test,learningrate = 1, numiterations = 500)