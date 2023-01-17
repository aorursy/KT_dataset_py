""""@author: shyam"""





#Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Salary_Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

####Fitting Data to the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
####Predicting the values

y_pred = regressor.predict(X_test)
####Visualizing the training set results

plt.scatter(X_train, y_train , color = 'red')

plt.plot(X_train,regressor.predict(X_train) , color = 'blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel('Experience')

plt.ylabel('Salary')

plt.show()
####Visualizing the test set results

plt.scatter(X_test, y_test , color = 'red')

plt.plot(X_train,regressor.predict(X_train) , color = 'blue')

plt.title('Salary vs Experience (Test Set)')

plt.xlabel('Experience')

plt.ylabel('Salary')

plt.show()
####Variance score of the model

from sklearn.metrics import explained_variance_score

vs = explained_variance_score(y_test,y_pred)

print(vs)