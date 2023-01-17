import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Salary_Data.csv')

features = ['YearsExperience']

# X = dataset.iloc[:, 0].values

X = dataset[features]

y = dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.title('Salary vs Years of Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.scatter(X_train, y_train, color='red')

plt.scatter(X_test, y_test, color='green')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.show()
y_pred