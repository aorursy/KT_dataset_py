# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/breast-cancer-wisconsin-data"))
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head()
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(['diagnosis'],axis=1)

print(y.shape,x_data.shape)
x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x=x.values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)







print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def sigmoid(z):

    return  1 / (1+np.exp(-z))
def costFunctionRegulize(theta, X, y, Lambda):

    m=len(y)

    y=y[:,np.newaxis]

    predictions = sigmoid(X @ theta)

    error = np.array((-y * np.log(predictions) - ((1-y)*np.log(1 - predictions))))

    cost = (1/m) * sum(error)

    regCost = cost + Lambda/(2*m) * sum(theta**2)

    j_0 = 1/m * (X.transpose() @ (predictions - y))[0]

    j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)*theta[1:]

    grad = np.vstack((j_0[:,np.newaxis],j_1))

    return cost[0], grad
def gradientDescent(X,y,theta,alpha,numIter,Lambda):

    m = len(y)

    costs = []

    for i in range(numIter):

        cost ,grad = costFunctionRegulize(theta,X,y,Lambda)

        theta -= alpha*grad

        costs.append(cost)

    return theta, costs
theta = np.zeros((x_train.shape[1],1))

Lambda = 0.2

alpha = 1

theta , costs = gradientDescent(x_train,y_train,theta,alpha,1500,Lambda)

plt.plot(costs)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")
def predict(theta,X):

    # X is a input for forward propagation

    z = sigmoid(X @ theta)

    Y_prediction = np.zeros((1,X.shape[0]))

    # if z is bigger than 0.5, our prediction is sign one (y=1),

    # if z is smaller than 0.5, our prediction is sign zero (y=0),

    for i in range(z.shape[0]):

        if z[i,0] < 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
y_prediction_train = predict(theta,x_train)

y_prediction_test = predict(theta,x_test)



print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
# sklearn

from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test)))

print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train)))