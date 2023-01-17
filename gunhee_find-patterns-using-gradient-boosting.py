import numpy as np

import pandas as pd

from collections import Counter

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df = pd.read_csv('../input/Daegu_Real_Estate_data.csv')
df.info()
df.isnull().sum()
Counter(df['TimeToSubway'])
Counter(df['TimeToBusStop'])
subway_mapping = {'0-5min': 4, '5min~10min': 3, '10min~15min': 2, '15min~20min': 1, 'no_bus_stop_nearby': 0}

bus_mapping = {'0~5min': 2, '5min~10min': 1, '10min~15min': 0}
df['TimeToSubway'] = df['TimeToSubway'].map(subway_mapping)

df['TimeToBusStop'] = df['TimeToBusStop'].map(bus_mapping)
fig, ax = plt.subplots(figsize=(12,10))

corr = df.corr()

sns.heatmap(corr, cmap="YlGnBu")
corr.iloc[0]
df['SalePrice'].skew()
# select numeric features

features = df.dtypes[df.dtypes != "object"].index



# make new dataframe

df = df[features]
df.shape
from sklearn import preprocessing



X = df.iloc[:, 1:].values

y= df.iloc[:, 0].values



stdsc = preprocessing.StandardScaler()

X_std = stdsc.fit_transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)
from sklearn import linear_model



# fit linear model

lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)
coef = lr.coef_

lr.coef_
coef_df = pd.Series(coef, index=df.columns[1:])
plt.rcParams['figure.figsize'] = (8, 10)

coef_df.sort_values().plot(kind = "barh")

plt.title("Coefficients Regression Model")

plt.xlabel("Coefficient")
print ("Root Mean squared error : %.3f" %(np.mean((lr.predict(X_test) - y_test)**2))**0.5)

print('Variance score: %.3f' % lr.score(X_test, y_test))
plt.scatter(lr.predict(X_test), y_test)



plt.xlabel('Predicted price')

plt.ylabel('Actual price')
# residual plot

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Train data')

plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc='upper left')

plt.hlines(y=0 ,xmin=0, xmax=550000, lw=2, color='red')

plt.show()
print("RMSE train: %.2f" % mean_squared_error(y_train, y_train_pred)**0.5)

print("RMSE test: %.2f"  % mean_squared_error(y_test, y_test_pred)**0.5)
plt.figure(figsize=(20,8))

sns.distplot(df['SalePrice'])

df['SalePrice'].describe()
sns.boxplot(y=df['SalePrice'])
df.sort_values(by='SalePrice').loc[df['SalePrice']>510000]
adj_df = df.drop(df.loc[df['SalePrice']>510000].index, axis=0)
X = adj_df.iloc[:, 1:].values

y= adj_df.iloc[:, 0].values



stdsc = preprocessing.StandardScaler()

X_std = stdsc.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)



# refit linear model

slr = linear_model.LinearRegression()

slr.fit(X_train, y_train)



coef = slr.coef_

slr.coef_

adj_coef_df = pd.Series(coef, index=adj_df.columns[1:])
print ("Root Mean squared error : %.2f" %(np.mean((slr.predict(X_test) - y_test)**2))**0.5)

print('Variance score: %.2f' % slr.score(X_test, y_test))
# compare

fig, ax = plt.subplots(nrows=1, ncols=2)



plt.subplot(1, 2, 1)

plt.rcParams['figure.figsize'] = (8, 10)

coef_df.sort_values().plot(kind = "barh")

plt.title("Coefficients Regression Model(original)")

plt.xlabel("Coefficient")





plt.subplot(1, 2, 2)

plt.rcParams['figure.figsize'] = (8, 10)

adj_coef_df.sort_values().plot(kind = "barh")

plt.title("Coefficients Regression Model(no outliers)")

plt.xlabel("Coefficient")



plt.subplots_adjust(wspace=1, right=1)
adj_df.drop(['N_FacilitiesNearBy(Total)', 'N_SchoolNearBy(High)', 

             'N_SchoolNearBy(Middle)', 'N_SchoolNearBy(Total)', 'N_Parkinglot(Ground)'], axis=1, inplace=True)
df.columns
sns.stripplot(x='N_elevators', y='N_APT', data=adj_df)
fig, ax = plt.subplots(figsize=(15,8))

sns.stripplot(x='N_Parkinglot(Basement)', y='N_APT', data=adj_df)
# delete more features

adj_df.drop(['N_elevators', 'N_Parkinglot(Basement)', 'MonthSold'], axis=1, inplace=True)
X = adj_df.iloc[:, 1:].values

y= adj_df.iloc[:, 0].values



stdsc = preprocessing.StandardScaler()

X_std = stdsc.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)



# fit linear model

slr = linear_model.LinearRegression()

slr.fit(X_train, y_train)



coef = slr.coef_

slr.coef_

adj_coef_df = pd.Series(coef, index=adj_df.columns[1:])
plt.rcParams['figure.figsize'] = (8, 10)

adj_coef_df.sort_values().plot(kind = "barh")

plt.title("Coefficients Regression Model(no outliers)")

plt.xlabel("Coefficient")
plt.scatter(slr.predict(X_test), y_test)



plt.xlabel('Prediction price')

plt.ylabel('Actual price')
print ("Root Mean squared error : %.3f" %(np.mean((slr.predict(X_test) - y_test)**2))**0.5)

print('Variance score: %.3f' % slr.score(X_test, y_test))
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>

#

# License: BSD 3 clause

from sklearn import ensemble





# #############################################################################

# Load data



X = adj_df.iloc[:, 1:].values

y= adj_df.iloc[:, 0].values



stdsc = preprocessing.StandardScaler()

X_std = stdsc.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)





# #############################################################################

# Fit regression model

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

rmse = mean_squared_error(y_test, clf.predict(X_test))**0.5

print("RMSE: %.3f" % rmse)

# #############################################################################

# Plot training deviance



# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')

# #############################################################################

# Plot feature importance

feature_importance = clf.feature_importances_



# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())



sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplots(figsize=(15, 10))

plt.subplot(1, 2, 2)



# exclude target variable in data frame

adj_df.drop('SalePrice', axis=1, inplace=True)



plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, adj_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
plt.scatter(clf.predict(X_test), y_test)



plt.xlabel('Prediction price')

plt.ylabel('Actual price')
# residual plot

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)



plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Train data')

plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc='upper left')

plt.hlines(y=0 ,xmin=0, xmax=500000, lw=2, color='red')

plt.show()