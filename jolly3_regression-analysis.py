import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
missing_values = ["n/a", "na", "--", "0"]

df = pd.read_csv('../input/Assignment1_50_Startups.csv', na_values = missing_values)
df.info()

df.head()
print (df.isnull().sum())

df[df['R&D Spend'].isnull() | df['Marketing Spend'].isnull()]
rd_median = df['R&D Spend'].median()

print(rd_median)

df['R&D Spend'].fillna(rd_median, inplace=True)
ms_median = df['Marketing Spend'].median()

print(ms_median)

df['Marketing Spend'].fillna(ms_median, inplace=True)
print (df.isnull().sum())
df['State'].value_counts()
df_dummies = pd.get_dummies(df['State'])

df_dummies.head()
df = pd.concat([df , df_dummies], axis=1)

df.columns
df.head()
x_features = ['R&D Spend', 'Administration', 'Marketing Spend', 'California', 'Florida']

x_df = df[x_features]

y_df = df['Profit']

X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)
m = len(Y_train)

n = len(X_train.columns)

weights = np.zeros((n+1,1))
X_train = (X_train - np.mean(X_train))/np.std(X_train)

X_test = (X_test - np.mean(X_test))/np.std(X_test)

X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)

Y_train = Y_train[:,np.newaxis]

Y_test = Y_test[:,np.newaxis]
X_train.head()
def gradientDescentMulti(X, y, weights, alpha, iterations):

    m = len(y)

#     print(X.shape)

#     print(y.shape)

#     print(weights.shape)

    for _ in range(iterations):

        temp = np.dot(X, weights) - y

        temp = np.dot(X.T, temp)

        weights = weights - (alpha/m) * temp

    return weights
def ridgegradientDescentMulti(X, y, weights, alpha, iterations, lamda):

    m = len(y)

#     print(X.shape)

#     print(y.shape)

#     print(weights.shape)

    for _ in range(iterations):

        temp = np.dot(X, weights) - y

        temp = np.dot(X.T, temp)

        weights = (weights*(1-alpha*(lamda/m))) - ((alpha/m) * temp)

    return weights
def lassogradientDescentMulti(X, y, weights, alpha, iterations, lamda):

    m = len(y)

#     print(X.shape)

#     print(y.shape)

#     print(weights.shape)

    for _ in range(iterations):

        temp = np.dot(X, weights) - y

        temp = np.dot(X.T, temp)

        weights = weights - ((alpha/m) * temp) - ((alpha/(2*m)) * lamda * weights/abs(weights))

    return weights
def GD_predict(X, weights):

    return np.dot(X, weights)
alpha = 0.01

num_iters = 5000

lamda = 0.05
weights = gradientDescentMulti(X_train, Y_train, weights, alpha, num_iters)

ridge_weights = ridgegradientDescentMulti(X_train, Y_train, weights, alpha, num_iters, lamda)

lasso_weights = lassogradientDescentMulti(X_train, Y_train, weights, alpha, num_iters , lamda)
Y_test[2]
GD_predict(X_test.iloc[2], weights)
GD_predict(X_test.iloc[2], ridge_weights)
GD_predict(X_test.iloc[2], lasso_weights)
Y_predicted = [GD_predict(x, weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
Y_predicted = [GD_predict(x, ridge_weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
Y_predicted = [GD_predict(x, lasso_weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
alpha = 0.005

num_iters = 5000

lamda = 0.05
weights = gradientDescentMulti(X_train, Y_train, weights, alpha, num_iters)

ridge_weights = ridgegradientDescentMulti(X_train, Y_train, weights, alpha, num_iters, lamda)

lasso_weights = lassogradientDescentMulti(X_train, Y_train, weights, alpha, num_iters , lamda)
Y_predicted = [GD_predict(x, weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
Y_predicted = [GD_predict(x, ridge_weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
Y_predicted = [GD_predict(x, lasso_weights) for x in X_test.values]

print("MSE",mean_squared_error(Y_test, Y_predicted))

plt.xlabel("Tested Profit")

plt.ylabel("Actual Profit")

plt.scatter(Y_test,Y_predicted)
df.describe()
sns.countplot(df['State'])

# sns.set_palette("GnBu_d")

# sns.set_style('whitegrid')

# sns.jointplot(x='R&D Spend',y='Profit',data=df,kind='scatter')
sns.pairplot(df)
