# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pandas as pd

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import datasets

from sklearn.linear_model import LogisticRegression

import numpy as np # linear algebra

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
iris = datasets.load_iris()

data = pd.DataFrame(iris.data, columns=['sepal_width', 'sepal_length','petal_width','petal_length'])

data['class'] = iris.target

ax = sns.scatterplot(x="sepal_width", y="sepal_length", data=data,hue="class",palette='Set1')
# Take first two attributes for classification

X1 = iris.data[:, :2]

#Convert to two class problem

y1 = (iris.target != 0) * 1
learning_rate=0.01

num_iterations=100000

threshold=0.5

    

#Create intercepts for equation with 1

intercept = np.ones((X1.shape[0], 1))

X1 = np.concatenate((intercept, X1), axis=1)



#Define sigmoid function

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



#Define loss function

def loss(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    

# weights initialization

theta1 = np.zeros(X1.shape[1])

        

for i in range(num_iterations):

    z = np.dot(X1, theta1)

    h = sigmoid(z)

    gradient = np.dot(X1.T, (h - y1)) / y1.size

    theta1 -= learning_rate * gradient



    if(i % 10000 == 0):

        z = np.dot(X1, theta1)

        h = sigmoid(z)

        print(f'loss: {loss(h, y1)} \t')

    

def predict_prob(X):

    return sigmoid(np.dot(X1, theta1))



def predict(X1, threshold):

    return predict_prob(X1) >= threshold
preds = predict(X1, threshold)

# accuracy

(preds == y1).mean()
# Let's check the coefficients

theta1
model = LogisticRegression(C=1e20)

%time model.fit(X1, y1)

preds = model.predict(X1)

# accuracy

(preds == y1).mean()
model.intercept_, model.coef_
#Function to Load and visualize data



# imports

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# load the data from the file

data = pd.read_csv("../input/Logistic_regression_practice.csv")



# X = feature values, all the columns except the last column

X = data.iloc[:, :-1]



# y = target values, last column of the data frame

y = data.iloc[:, -1]



# filter out the applicants that got admitted

admitted = data.loc[y == 1]



# filter out the applicants that din't get admission

not_admitted = data.loc[y == 0]



# plots

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')

plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

plt.legend()

plt.show()
learning_rate=0.001

num_iterations=100000

threshold=0.5

    

#Create intercepts for equation with 1

intercept = np.ones((X.shape[0], 1))

X = np.concatenate((intercept, X), axis=1)



#Define sigmoid function

def sigmoid(z):

    return 1 / (1 + np.exp(-z))



#Define loss function

def loss(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    

# weights initialization

theta = np.zeros(X.shape[1])

        

for i in range(num_iterations):

    z = np.dot(X, theta)

    h = sigmoid(z)

    gradient = np.dot(X.T, (h - y)) / y.size

    theta -= learning_rate * gradient



    if(i % 10000 == 0):

        z = np.dot(X, theta)

        h = sigmoid(z)

        print(f'loss: {loss(h, y)} \t')

    

def predict_prob(X):

    return sigmoid(np.dot(X, theta))



def predict(X, threshold):

    return predict_prob(X) >= threshold
preds = predict(X, threshold)

# accuracy

(preds == y).mean()
theta
model = LogisticRegression(C=1e20)

%time model.fit(X, y)

preds = model.predict(X)

# accuracy

(preds == y).mean()