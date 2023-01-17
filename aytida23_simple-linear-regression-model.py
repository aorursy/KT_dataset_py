"""
This is a code for the Simple Linear Regression model for predicting the values
from a test set by learning the correlation between the training set data. This
is a machine learning model called Simple Linear Regression.
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('../input/Salary_Data.csv')
x = dataset.iloc[:,[0]]
x
y = dataset.iloc[:,[1]]
y
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(x_test)
y_pred
y_test
# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'cyan')
plt.plot(x_train, regressor.predict(x_train), color = 'brown')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'cyan')
plt.plot(x_test, regressor.predict(x_test), color = 'brown')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
