# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory

# print all the file/directories present at the path

import os

print(os.listdir("../input/"))
# importing the dataset

dataset = pd.read_csv('../input/Salary_Data.csv')
dataset.head()
dataset.info()
# checking if any null data is present

dataset.isnull().sum()
# Creating matrix of features as X and dependant variable vector as Y

X = dataset.iloc[:,0:1].values

Y = dataset.iloc[:,-1].values
# split into train & test data 

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
# Creating simple linear regression object & fit to train set 

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train,Y_train)
# Visualizing the performance of the model ---- on training set

from matplotlib import pyplot

pyplot.plot(X_train,reg.predict(X_train),color='red')

pyplot.scatter(X_train,Y_train,color='blue')

pyplot.title('Salary vs Experience - Best Fit Line')

# Visualizing the performance of the model ---- on test set

from matplotlib import pyplot

pyplot.plot(X_test,reg.predict(X_test),color='red')

pyplot.scatter(X_test,Y_test,color='blue')

pyplot.title('Salary vs Experience - Best Fit Line')