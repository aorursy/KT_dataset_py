import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
dataset = pd.read_csv('../input/Baltimore_Salary.csv')

dataset.info()
dataset.head(10)
X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(y_test)
print(y_pred)
plt.scatter(X_train,y_train, color = 'red')

plt.plot(X_train, reg.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel('Year of Experience')

plt.ylabel('Salary')

plt.show()
plt.scatter(X_test,y_test, color = 'red')

plt.plot(X_train, reg.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training Set)')

plt.xlabel('Year of Experience')

plt.ylabel('Salary')

plt.show()