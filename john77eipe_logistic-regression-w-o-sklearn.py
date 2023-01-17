# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/titanic/test.csv")

data = pd.read_csv("../input/titanic/train.csv")
data_train_x = ['Sex']



data_train = pd.get_dummies(data[data_train_x])
fig, ax = plt.subplots(1, 2, figsize = (14, 5))

data["Sex"].value_counts().plot.bar(color = "skyblue", ax = ax[0])

ax[0].set_title("Number Of Passengers By Sex")

ax[0].set_ylabel("Population")

sns.countplot("Sex", hue = "Survived", data = data, ax = ax[1])

ax[1].set_title("Sex: Survived vs Dead")

plt.show()
#setting the matrixes

X = data_train.iloc[:].values
noOfTrainEx = X.shape[0] # no of training examples

print("noOfTrainEx: ",noOfTrainEx)

noOfWeights = X.shape[1]+1 # no of features+1 => weights

print("noOfWeights: ", noOfWeights)
ones = np.ones([noOfTrainEx, 1]) # create a array containing only ones 

X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix

theta = np.ones((1, noOfWeights)) #np.array([[1.0, 1.0]])
y = data['Survived'].values.reshape(-1,1) # create the y matrix
print(X.shape)

print(theta.shape)

print(y.shape)
def sigmoid(z):

        return 1 / (1 + np.exp(-z))

def computeCost(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
#set hyper parameters

alpha = 0.01

iters = 3000
## Gradient Descent funtion

def gradientDescent(X, y, theta, alpha, iters):

    cost = np.zeros(iters)

    for i in range(iters):

        z = X @ theta.T

        h = sigmoid(z)

        theta = theta - (alpha/len(X)) * np.sum((h - y) * X, axis=0)

        cost[i] = computeCost(h, y)

        if i % 100 == 0: # just look at cost every ten loops for debugging

            print(i, 'iteration, cost:', cost[i])

    return (theta, cost)
g, cost = gradientDescent(X, y, theta, alpha, iters)  
print(g)
#plot the cost

fig, ax = plt.subplots()  

ax.plot(np.arange(iters), cost, 'r')  

ax.set_xlabel('Iterations')  

ax.set_ylabel('Cost')  

ax.set_title('Error vs. Training Epoch')
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1e20)

%time model.fit(X, y)



preds = model.predict(X)

# accuracy

print ("Accuracy: ", (preds == y).mean())
model.intercept_, model.coef_