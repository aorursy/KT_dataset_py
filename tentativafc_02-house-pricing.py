# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
df_train_original = pd.read_csv('../input/train.csv')

df_test_original = pd.read_csv('../input/test.csv')
df_train_original.columns
df_train_original.head()
df_test_original.head()
cols = ['LotArea', 'YearBuilt', 'GarageArea', 'FullBath']

cols_test = cols + ['Id']

cols_train = cols + ['Id', 'SalePrice']
df_train = df_train_original[cols_train].copy()

df_test = df_test_original[cols_test].copy()

print("Nan of trains", df_train[df_train.isnull().any(axis=1)])

print("Nan of tests", df_test[df_test.isnull().any(axis=1)])
df_test = df_test.fillna(df_test.median())

print("Nan of tests", df_test[df_test.isnull().any(axis=1)])
df_test
cols_with_dependent = ['SalePrice'] + cols



num_lines = len(cols_with_dependent)

fig, ax = plt.subplots(nrows=num_lines, ncols=1, figsize=(20, num_lines*5))



for index, col in enumerate(cols_with_dependent):

    ax[index].hist(df_train[col])

    ax[index].title.set_text(f"Histogram of {col}")

    ax[index].set_xlabel(f"{col}")

    ax[index].set_ylabel("Frequency")



plt.show()
boxplot = df_train.boxplot(column=['SalePrice'])
df_train_without_outliers = df_train[df_train['SalePrice'] < 400000].copy()
df_train['L_SalePrice'] = np.log(df_train['SalePrice'])

df_train_without_outliers['L_SalePrice'] = np.log(df_train_without_outliers['SalePrice'])

plt.hist(df_train['L_SalePrice'])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0].boxplot(df_train['L_SalePrice'])

ax[0].title.set_text("Histogram Log")

ax[0].set_ylabel("Log Sale' Price")



ax[1].boxplot(df_train_without_outliers['L_SalePrice'])

ax[1].title.set_text("Histogram Log Without outliers")

ax[1].set_ylabel("Log Sale' Price")

# plt.plot(df_train_original['YearBuilt'], df_train_original['SalePrice'],  'o')

# plt.title("Year Build X Sale's Price")

# plt.xlabel('Year Build')

# plt.ylabel("Sale's Price")



num_lines = len(cols)

fig, ax = plt.subplots(nrows=num_lines, ncols=2, figsize=(20, num_lines * num_lines))



for index, col in enumerate(cols):

    ax[index, 0].scatter(df_train[col], df_train['SalePrice'])

    ax[index, 0].title.set_text(f"{col} X Sale's Price")

    ax[index, 0].set_xlabel(f"{col}")

    ax[index, 0].set_ylabel("Sale's Price")

    

    ax[index, 1].scatter(df_train[col], df_train['L_SalePrice'])

    ax[index, 1].title.set_text(f"{col} X Log Sale's Price")

    ax[index, 1].set_xlabel(f"{col}")

    ax[index, 1].set_ylabel("Log Sale's Price")



plt.show()
y = df_train['SalePrice']

X = df_train[cols]

reg = LinearRegression().fit(X, y)

X_test = df_test[cols].copy()

print("Score", reg.score(X, y))

y_hat = reg.predict(X_test)
# y = df_train_without_outliers['L_SalePrice']

# X = df_train_without_outliers[cols]

# reg = LinearRegression().fit(X, y)

# X_test = df_test[cols]

# print("Score", reg.score(X, y))

# reg.predict(X_test)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_hat})

my_submission.to_csv('submission.csv', index=False)