import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
train_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
features_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
stores_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
train_df.head()
train_df.columns
train_df.shape
train_df.info()
train_df.describe()
train_df.isnull()
train_df.isnull().sum()
test_df.head()
test_df.columns
test_df.shape
test_df.info()
test_df.describe()
test_df.isnull()
test_df.isnull().sum()
features_df.head()
features_df.columns
features_df.shape
features_df.info()
features_df.describe()
features_df.isnull()
features_df.isnull().sum()
stores_df.head()
stores_df.columns
stores_df.shape
stores_df.info()
stores_df.describe()
stores_df.isnull()
stores_df.isnull().sum()
dataset = features_df.merge(stores_df, how= 'inner', on = 'Store')
dataset.head()
train_df.head()
dataset.info()
# Handling The Datetime

from datetime import datetime

dataset['Date'] = pd.to_datetime(dataset['Date'])
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
dataset['week'] = dataset.Date.dt.isocalendar().week
dataset['year'] = dataset.Date.dt.isocalendar().year
dataset.head()
train_df.head()
# Merge dataset with train_df

train_df = train_df.merge(dataset, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
# Merge dataset with test_df

test_df = test_df.merge(dataset, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
train_df.head()
test_df.head()
def scatter(train_df, column):
    plt.figure()
    plt.scatter(train_df[column], train_df['Weekly_Sales'])
    plt.ylabel('Weekly_Sales')
    plt.xlabel(column)
scatter(train_df, 'Store')
scatter(train_df, 'Dept')
scatter(train_df, 'IsHoliday')
scatter(train_df, 'Temperature')
scatter(train_df, 'Fuel_Price')
scatter(train_df, 'CPI')
scatter(train_df, 'Unemployment')
scatter(train_df, 'Type')
scatter(train_df, 'Size')
# Average Weekly Sales for the year 2010
weekly_sales_2010 = train_df[train_df['year'] ==  2010]['Weekly_Sales'].groupby(train_df['week']).mean()
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
# Average Weekly Sales for the year 2011
weekly_sales_2011 = train_df[train_df['year']== 2011]['Weekly_Sales'].groupby(train_df['week']).mean()
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
# Average Weekly Sales for the year 2012
weekly_sales_2012 = train_df[train_df['year']== 2012]['Weekly_Sales'].groupby(train_df['week']).mean()
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
# Plotting the above three plot together

plt.figure(figsize= (20, 10))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1,60, step= 1))
plt.title('Average Weekly Sales Per Year', fontsize = 20)
plt.xlabel('Week', fontsize = 16)
plt.ylabel('Sales', fontsize = 16)
plt.legend(['2010', '2011', '2012'], loc = 'best', fontsize = 16)
plt.show()
# Average Sales per Department
weekly_sales = train_df['Weekly_Sales'].groupby(train_df['Dept']).mean()
plt.figure(figsize= (25, 12))
sns.barplot(weekly_sales.index, weekly_sales.values)
plt.grid()
plt.title('Average Weekly Sales Per Department', fontsize = 20)
plt.xlabel('Department', fontsize = 16)
plt.ylabel('Sales', fontsize = 16)
plt.show()
# Average Sales per Store
weekly_sales = train_df['Weekly_Sales'].groupby(train_df['Store']).mean()
plt.figure(figsize= (25, 12))
sns.barplot(weekly_sales.index, weekly_sales.values)
plt.grid()
plt.title('Average Weekly Sales Per Department', fontsize = 20)
plt.xlabel('Store', fontsize = 16)
plt.ylabel('Sales', fontsize = 16)
plt.show()
sns.set(style = 'white')
corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype = np.bool))
fig, ax = plt.subplots(figsize= (25, 15))
cmap = sns.diverging_palette(220, 10, as_cmap= True)
sns.heatmap(corr, mask = mask, cmap = cmap, vmax = 0.3, center = 0, square = True, linewidth= 0.5, cbar_kws = {'shrink': 0.5}, annot = True)
# Dropping down the variables that have weak correlation
train_df.drop(columns = ['Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace = True)
test_df.drop(columns = ['Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], inplace = True)
train_df.head()
test_df.head()
# Splitting Data into Train and Test
X = train_df[['Store', 'Dept', 'IsHoliday','Size','week','year']]
y = train_df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train.info()
# Performing GridSearchCV on Ridge Regression
params = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
ridge_regressor = GridSearchCV(Ridge(), params, cv = 7, scoring = 'neg_mean_absolute_error', n_jobs = -1)
ridge_regressor.fit(X_train, y_train)
# Predicting train and test results
y_train_pred = ridge_regressor.predict(X_train)
y_test_pred = ridge_regressor.predict(X_test)
print("Train Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))
# Performing GridSearchCV on Lasso Regression
params = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
lasso_regressor = GridSearchCV(Lasso(), params ,cv = 15,scoring = 'neg_mean_absolute_error', n_jobs = -1)
lasso_regressor.fit(X_train, y_train)
# Predicting train and test results
y_train_pred = lasso_regressor.predict(X_train)
y_test_pred = lasso_regressor.predict(X_test)
print("Train Results for Lasso Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Lasso Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))
# Performing GridSearchCV on Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

depth = list(range(3,30))
param_grid = dict(max_depth = depth)
tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 10)
tree.fit(X_train,y_train)
# Predicting train and test results
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
print("Train Results for Decision Tree Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Decision Tree Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))
# Performing RandomsearchCV on Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
tuned_params = {'n_estimators': [100, 200], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}  
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 3, scoring = 'neg_mean_absolute_error', cv = 3, n_jobs = -1)
random_regressor.fit(X_train, y_train)
# Predicting train and test results
y_train_pred = random_regressor.predict(X_train)
y_test_pred = random_regressor.predict(X_test)
print("Train Results for Random Forest Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Random Forest Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))
prediction = lasso_regressor.predict(test_df[['Store', 'Dept', 'IsHoliday','Size','week','year']])
prediction
df.head()
df.Weekly_Sales = prediction
df.to_csv('weekly_sales.csv', index = False)
