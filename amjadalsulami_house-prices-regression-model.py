# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# linear algebra

# data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load the data

df_train= pd.read_csv('../input/train.csv')

df_test = pd.read_csv("../input/test.csv")
df_train.columns
# Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']



# Drop the 'Id' column because it is unnecessary for the prediction.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)
# statistics summary(descreptive)

df_train['SalePrice'].describe()
#histogram

sns.distplot(df_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#correlation matrix

corr_matrix = df_train.corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_matrix, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

correlation_matrix = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

heatmap = sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#most correlated features with the saleprice 

most_corr = pd.DataFrame(cols)

most_corr.columns = ['Most Correlated Features']

most_corr
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF','FullBath', 'YearBuilt','1stFlrSF','TotRmsAbvGrd']

sns.pairplot(df_train[cols], height = 2.5)

plt.show();
#Investigate on missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(['Utilities'], axis=1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #check no missing data
df_train.describe()
#convert categorical variable into dummy variables

df_train = pd.get_dummies(df_train)
# First i will use the features that i got from the heatmap

cols_name=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','1stFlrSF','TotRmsAbvGrd']

X=df_train[cols_name]

y=df_train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# build a linear regression model

model = LinearRegression()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))



# model evaluation with RMSE

print(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
# define X and y 

X = df_train.drop('SalePrice', axis=1)

y=df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)

print(model.score(X_test, y_test))

# model evaluation with RMSE

np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
# Lasso regression

# try alpha=75 and examine coefficients

lassoreg = Lasso(alpha=80, normalize=True)

lassoreg.fit(X_train, y_train)

print(lassoreg.score(X_test, y_test))

y_pred = lassoreg.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))