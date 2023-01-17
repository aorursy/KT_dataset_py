# read the data and set the datetime as the index

import pandas as pd

#url = 'https://raw.githubusercontent.com/upxacademy/ML_with_Python/master/Datasets/bikeshare.csv?token=AYxzdiGnjM610dBT7PuwUnUNOmm3bGcvks5ZFDyLwA%3D%3D'

#bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)

bikes = pd.read_csv("../input/train.csv",  index_col='datetime',parse_dates=True)

bikes_test = pd.read_csv("../input/test.csv", index_col='datetime',parse_dates=True)
# len(bikes)

print(bikes.head())

bikes_test.head()

# "count" is a method, so it's best to name that column something else

bikes.rename(columns={'count':'total'}, inplace=True)
bikes.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
# Pandas scatter plot

bikes.plot(kind='scatter', x='temp', y='total', alpha=0.9)
# Seaborn scatter plot with regression line

sns.lmplot(x='temp', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.9})
# create X and y

feature_cols = ['temp']

X_test = bikes_test[feature_cols]

X = bikes[feature_cols]

y = bikes.total
# import, instantiate, fit

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X, y)

#linreg.score(X,y)
# print the coefficients

print (linreg.intercept_)

print (linreg.coef_)
# manually calculate the prediction

linreg.intercept_ + linreg.coef_*25
# use the predict method

linreg.predict(25)



# create a new column for Fahrenheit temperature

bikes['temp_F'] = bikes.temp * 1.8 + 32

bikes.head()
# Seaborn scatter plot with regression line

sns.lmplot(x='temp_F', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})
# create X and y

feature_cols = ['temp_F']

X = bikes[feature_cols]

y = bikes.total



# instantiate and fit

linreg = LinearRegression()

linreg.fit(X, y)



# print the coefficients

print (linreg.intercept_)

print (linreg.coef_)
# convert 25 degrees Celsius to Fahrenheit

25 * 1.8 + 32
# predict rentals for 77 degrees Fahrenheit

linreg.predict(77)
# remove the temp_F column

bikes.drop('temp_F', axis=1, inplace=True)
##have to bring it to uniform scale
# explore more features

feature_cols = ['temp', 'season', 'weather', 'humidity']
# multiple scatter plots in Seaborn

sns.pairplot(bikes, x_vars=feature_cols, y_vars='total', kind='reg')
# multiple scatter plots in Pandas

fig, axs = plt.subplots(1, len(feature_cols), sharey=True)

for index, feature in enumerate(feature_cols):

    bikes.plot(kind='scatter', x=feature, y='total', ax=axs[index], figsize=(16, 3))

    plt.plot()
    #correlation coff.

    bikes.corr()
# box plot of rentals, grouped by season

bikes.boxplot(column='total', by='season')
# line plot of rentals

bikes.total.plot()
# create a list of features

feature_cols = ['temp', 'season', 'weather', 'humidity']
# create X and y

X = bikes[feature_cols]

y = bikes.total



# instantiate and fit

linreg = LinearRegression()

linreg.fit(X, y)



# print the coefficients

print (linreg.intercept_)

print (linreg.coef_)
# pair the feature names with the coefficients

list(zip(feature_cols, linreg.coef_))
# example true and predicted response values

true = [10, 7, 5, 5]

pred = [8, 6, 5, 10]
# calculate these metrics by hand!

from sklearn import metrics

import numpy as np

print ('MAE:', metrics.mean_absolute_error(true, pred))

print ('MSE:', metrics.mean_squared_error(true, pred))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(true, pred)))
# same true values as above

true = [10, 7, 5, 5]



# new set of predicted values

pred = [10, 7, 5, 13]



# MAE is the same as before

print ('MAE:', metrics.mean_absolute_error(true, pred))



# MSE and RMSE are larger than before

print ('MSE:', metrics.mean_squared_error(true, pred))

print ('RMSE:', np.sqrt(metrics.mean_squared_error(true, pred)))
from sklearn.model_selection import train_test_split

import numpy as np



# define a function that accepts a list of features and returns testing RMSE

def train_test_rmse(feature_cols):

    X = bikes[feature_cols]

    y = bikes.total

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    linreg = LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)

    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# compare different sets of features

print (train_test_rmse(['temp', 'season', 'weather', 'humidity']))

print (train_test_rmse(['temp', 'season', 'weather']))

print (train_test_rmse(['temp', 'season', 'humidity']))
# using these as features is not allowed!

print (train_test_rmse(['casual', 'registered']))
# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)



# create a NumPy array with the same shape as y_test

y_null = np.zeros_like(y_test, dtype=float)



# fill the array with the mean value of y_test

y_null.fill(y_test.mean())

y_null
# compute null RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_null))
# create dummy variables

season_dummies = pd.get_dummies(bikes.season, prefix='season')



# print 5 random rows

season_dummies.sample(n=5, random_state=1)
# drop the first column

season_dummies.drop(season_dummies.columns[0], axis=1, inplace=True)



# print 5 random rows

season_dummies.sample(n=5, random_state=1)
# concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns)

bikes = pd.concat([bikes, season_dummies], axis=1)



# print 5 random rows

bikes.sample(n=5, random_state=1)
# include dummy variables for season in the model

feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']

X = bikes[feature_cols]

y = bikes.total

linreg = LinearRegression()

linreg.fit(X, y)

list(zip(feature_cols, linreg.coef_))

# compare original season variable with dummy variables

print (train_test_rmse(['temp', 'season', 'humidity']))

print (train_test_rmse(['temp', 'season_2', 'season_3', 'season_4', 'humidity']))
# hour as a numeric feature

bikes['hour'] = bikes.index.hour

#bikes.index.to_native_types

#bikes.dtypes

#bikes.to_datetime(raw_data['index'], format='%Y%d%b:%H:%M:%S.%f')

#bikes.info()
bikes.head(2)
# hour as a categorical feature

hour_dummies = pd.get_dummies(bikes.hour, prefix='hour')

hour_dummies.drop(hour_dummies.columns[0], axis=1, inplace=True)

bikes = pd.concat([bikes, hour_dummies], axis=1)
# daytime as a categorical feature

bikes['daytime'] = ((bikes.hour > 6) & (bikes.hour < 21)).astype(int)
print (train_test_rmse(['hour']))

print (train_test_rmse(bikes.columns[bikes.columns.str.startswith('hour_')]))

print (train_test_rmse(['daytime']))