# Importing all the required libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Importing the dataset from your file directory where you have downloaded your csv file



dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

dataset.head()
dataset.shape
dataset.isnull().sum()
dataset.describe()
X = dataset.iloc[:, :-1]

Y = dataset.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=123)

print ("Training set size:", X_train.shape)

print ("Test set size:", X_test.shape)
regressor = LinearRegression()

regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
compare = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

compare.head(10)
df = compare.head(25)

df.plot(kind='bar',figsize=(10,8))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))