# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Import Libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import sklearn



#Importing DataSet 

dataset = pd.read_csv("../input/kc_house_data.csv")

space=dataset['sqft_living']

price=dataset['price']



x = np.array(space).reshape(-1, 1)

y = np.array(price)



#Splitting the data into Train and Test

from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)





#Fitting simple linear regression to the Training Set

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression()

regressor.fit(xtrain, ytrain)



#Predicting the prices

pred = regressor.predict(xtest)



#Visualizing the training Test Results 

plt.scatter(xtrain, ytrain, color= 'red')

plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')

plt.title ("Visuals for Training Dataset")

plt.xlabel("Space")

plt.ylabel("Price")

plt.show()



#Visualizing the Test Results 

plt.scatter(xtest, ytest, color= 'red')

plt.plot(xtest, regressor.predict(xtest), color = 'blue')

plt.title("Visuals for Test DataSet")

plt.xlabel("Space")

plt.ylabel("Price")

plt.show()
