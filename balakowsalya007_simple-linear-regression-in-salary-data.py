import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm
salaryData = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

salaryData.head()
salaryData.describe()
salaryData.shape
salaryData.isnull().sum()
X = salaryData.iloc[:,:-1]

y = salaryData.iloc[:,-1]
print(X)

print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 111, test_size = 0.3)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X, y, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test Set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary') 

plt.show()