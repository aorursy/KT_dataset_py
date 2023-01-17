#Importing libraries

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))
#Importing dataset

dataset = pd.read_csv("../input/04Salary_Data.csv")
#Independent and dependent variables

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:, 1].values
#Splitting training and testsets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state=0)
#Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression();

regressor.fit(X_train, y_train)

#Predicting the test set results

y_pred = regressor.predict(X_test)
#Visualising the training set results

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train,regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()



#Visualising the test set results

plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train,regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test Set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()


