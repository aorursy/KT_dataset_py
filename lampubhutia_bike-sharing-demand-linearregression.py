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
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
bikes=pd.read_csv("../input/bikeshare.csv", index_col='datetime', parse_dates=True)

bikes.head()
# "count" is a function, so to avoid  confusion we change the column name to total

bikes.rename(columns={'count':'total'}, inplace=True)
bikes_data=bikes.copy()
print(bikes_data.shape)
bikes_data.describe()
# To check Multicollinearity 



bikes_data.corr()
# scatter plot

a=sns.lmplot(x='temp', y='total', fit_reg=True, data=bikes_data, aspect=1.5, scatter_kws={'alpha':0.2})

# exploring more features

feature_cols = ['temp', 'season', 'weather', 'humidity']
# multiple scatter plots in Seaborn

sns.pairplot(bikes_data, x_vars=feature_cols, y_vars='total', kind='reg')
# box plot of rentals, grouped by season

bikes.boxplot(column='total', by='season')
# create dummy variables

season_dummies = pd.get_dummies(bikes_data.season, prefix='season')



# print 5 random rows

season_dummies.sample(n=5, random_state=12)
# drop the first column

season_dummies.drop(season_dummies.columns[0], axis=1, inplace=True)



# print 5 random rows

season_dummies.sample(n=5, random_state=12)
# concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns)

bikes_data = pd.concat([bikes_data, season_dummies], axis=1)



# print 5 random rows

bikes_data.sample(n=5, random_state=12)
# include dummy variables for season in the model

feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']

X = bikes_data[feature_cols]  # input or independent variable 

y = bikes_data.total          # Output or Dependent variable

linreg = LinearRegression()

linreg.fit(X, y)

list(zip(feature_cols, linreg.coef_))
# splitting the data into training and test data.



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=12)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Buliding the Linear model with the algorithm

lin_reg=LinearRegression()

model=lin_reg.fit(X_train,y_train)
# feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity'] #Input or independent variable

print(model.intercept_)

print (model.coef_)
## Predicting the x_test with the model

predicted=model.predict(X_test)
print ('MAE:', metrics.mean_absolute_error(y_test, predicted))

print ('MSE:', metrics.mean_squared_error(y_test, predicted))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
# ** To measure accuracy of model the model generated RMSE value has to be lower than null RMSE** 



#Compute null RMSE

# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=12)



# create a NumPy array with the same shape as y_test

y_null = np.zeros_like(y_test, dtype=float)



# fill the array with the mean value of y_test

y_null.fill(y_test.mean())

y_null
print(y_test.shape)

print(y_null.shape)
# compute null RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_null))
# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(feature_cols):

    X = bikes_data[feature_cols]

    y = bikes_data.total

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# compare different sets of features

print (train_test_rmse(['temp', 'season', 'weather', 'humidity']))

print (train_test_rmse(['temp', 'season', 'weather']))

print (train_test_rmse(['temp', 'season', 'humidity']))

print (train_test_rmse(['temp', 'humidity']))

print (train_test_rmse(['temp', 'season_2', 'season_3', 'season_4','weather', 'humidity']))

print (train_test_rmse(['temp', 'season_2', 'season_3', 'season_4','weather']))

print (train_test_rmse(['temp', 'season_2', 'season_3', 'season_4', 'humidity']))
bikes_data['hour']=bikes_data.index.hour
bikes_data.head()
# hour as a categorical feature

hour_dummies = pd.get_dummies(bikes_data.hour, prefix='hour')

hour_dummies.drop(hour_dummies.columns[0], axis=1, inplace=True)

bikes_data = pd.concat([bikes_data, hour_dummies], axis=1)

#hour_dummies

bikes_data.head()
# with hour.

sns.factorplot(x="hour",y="total",data=bikes_data,kind='bar',size=5,aspect=1.5)
# hour as a categorical feature

hour_dummies = pd.get_dummies(bikes_data.hour, prefix='hour')

hour_dummies.drop(hour_dummies.columns[0], axis=1, inplace=True)

bikes_data = pd.concat([bikes_data, hour_dummies], axis=1)

#hour_dummies

bikes_data.head()
# daytime as a categorical feature

bikes_data['daytime'] = ((bikes_data.hour > 6) & (bikes_data.hour < 21)).astype(int)

bikes_data.tail()
print (train_test_rmse(['hour']))

print (train_test_rmse(bikes_data.columns[bikes_data.columns.str.startswith('hour_')]))

print (train_test_rmse(['daytime']))
print('END')