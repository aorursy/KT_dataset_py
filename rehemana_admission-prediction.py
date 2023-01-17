import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
df = pd.read_csv(r'../input/graduate-admissions/Admission_Predict.csv')

df2 = pd.read_csv(r'../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.shape
df2.shape
df.head(10)
df2.head(10)
scatter_matrix(df2.iloc[:, 1:], figsize=(15, 12))

plt.show()
# Correlation

plt.figure(figsize=(15, 10))

sns.heatmap(df2.iloc[:, 1:].corr(), annot = True, cmap='gray', vmin=0, vmax=1)
df2.isnull().sum()
df2.dtypes
df2.columns.values
y = np.asanyarray(df2.iloc[:, -1]) # 'Chance of Admit'

X = np.asanyarray(df2.loc[:,['GRE Score', 'TOEFL Score', 'CGPA']]) # continous valued features
import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

# standardize the feature values

X= preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn import linear_model

#

# Build the model

lr = linear_model.LinearRegression()

# Train it

lr.fit(X_train, y_train)

# Make the prediction

y_hat = lr.predict(X_test)

print("Residual sum of squares: %.4f" % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % lr.score(X_test, y_test))
# Check RMSE and R2_score

from sklearn.metrics import mean_squared_error, r2_score

test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_hat)))

test_set_r2 = r2_score(y_test, y_hat)

#

print(test_set_rmse)

print(test_set_r2)