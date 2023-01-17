# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt # Plotting
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the dataset
dataset = pd.read_csv('../input/Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values
X = X.reshape(-1,1)
y = y.reshape(-1,1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Imported linear regressor class then created an object from this called regressor. Then used one of the mehtods in the class called fit to fit the simple linear model to the training data set.
#This is the model. Here we created a machine 'regressor' to learn the correlation between the experience and salary. 
#Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred
y_test
#By comparing real salaries in y_test to predicted salary y_pred we can find the correlation between the experience and salary.
#Visualizing the Training set results 
#Plot employees of the company categorized by their number of years of experience by their salary
plt.scatter(X_train,y_train, color='red',)
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()
#Next we will plot the test set observation points by keeping the training set regression line as it is.
#Visualizing the Test Set results
plt.scatter(X_test,y_test, color='red',)
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()
#This is the final predicted machine learning model using simple linear regression. Cheers!