# Import the module 'LinearRegression' from sklearn

from sklearn.linear_model import LinearRegression
# Create an object of type LinearRegression

model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

model
import numpy as np

import pandas as pd

X = pd.read_csv('../input/retinopathy-dummy-data/X_data.csv')

Y = pd.read_csv('../input/retinopathy-dummy-data/y_data.csv')

print(X.shape)

X.head()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,6))

sns.distplot(X['Age'])

plt.title('Age Distribution')

plt.show()



plt.figure(figsize=(10,6))

sns.distplot(X['Systolic_BP'])

plt.title('Systolic_BP Distribution')

plt.show()



plt.figure(figsize=(10,6))

sns.distplot(X['Diastolic_BP'])

plt.title('Diastolic_BP Distribution')

plt.show()



plt.figure(figsize=(10,6))

sns.distplot(X['Cholesterol'])

plt.title('Cholesterol Distribution')

plt.show()
X['Age'].hist()
X['Systolic_BP'].hist()
X['Diastolic_BP'].hist()
# Fit the linear regression model

model.fit(X, Y)

model
# View the coefficients of the model

model.coef_
np.dot(X.values , model.coef_.T).squeeze()
sns.distplot(np.dot(X.values , model.coef_.T).squeeze())