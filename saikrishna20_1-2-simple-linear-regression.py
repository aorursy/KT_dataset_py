import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
dataset = pd.read_csv('../input/salary/Salary.csv')
X = dataset.iloc[:, :-1].values 
# .values - we have converted it to an array because our 
# models expects a 2 dimensional array to process
y = dataset.iloc[:, -1].values
dataset

X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
# these are the real values we have plotted as dots and is scattered
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plotting the X_train on x-axis and predicted values by using our model on the y-axis
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
# 1/3 i.e 33% of data we have splitted and stored as testing is plotted
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# the test data and the predicted values of test data are plotted
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()
print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)