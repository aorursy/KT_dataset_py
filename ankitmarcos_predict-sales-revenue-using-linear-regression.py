#Run this code cell in Google Co-Lab

!wget https://www.dropbox.com/s/hudkj000ffz4lkl/Advertising.csv
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('Advertising.csv')
df.head()
df.shape
df.describe()
df = df.drop('Unnamed: 0', axis=1)
df.head()
fig = px.histogram(df, x="sales")

fig.show()
fig = px.histogram(df, x="newspaper")

fig.show()
fig = px.histogram(df, x="radio")

fig.show()
fig = px.histogram(df, x="TV")

fig.show()
sns.pairplot(df, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg');

plt.show()
df.TV.corr(df.sales)
df.corr()
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True )

plt.show()
X = df[['TV']]

X.head()
# check the type and shape of X

print(type(X))

print(X.shape)
y = df.sales

y.head()
print(type(y))

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)
# print the intercept and coefficients

print(linreg.intercept_)

print(linreg.coef_)
# make predictions on the testing set

y_pred = linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))