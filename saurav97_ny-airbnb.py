# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor 

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(dirname + '/' + filename)

data.columns
data
# Checking for NULL Values present in the data

data.isnull().sum()
# Filling the NULL value in reviews per month by 0 

data['reviews_per_month'].fillna(0,inplace = True)



# Dropping the columns last_review as ~25% of the data is NULL

data.drop(['last_review'],axis=1,inplace=True)



# Dropping them due to lack of relevance in price prediction

data.drop(['name'],axis=1,inplace=True)

data.drop(['host_name'],axis=1,inplace=True)
# Printing the correlation matrix to see the importance of different variables in predicting the price

corr = data.corr(method='spearman')

plt.figure(figsize=(10,8))

sns.heatmap(corr, annot=True)
fig, axes = plt.subplots(1,2, figsize=(15,4))

sns.countplot(data['neighbourhood_group'], ax = axes[0])

sns.countplot(data['room_type'], ax = axes[1])

fig = plt.gcf()
fig, axes = plt.subplots(1,2, figsize=(16,4))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data, ax = axes[0])

sns.catplot(x="room_type", y="price", data=data, ax = axes[1])

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', data=data)

plt.show()
fig, axes = plt.subplots(1,2, figsize=(15,4))

sns.distplot(data['price'], ax = axes[0])

sns.distplot(np.log1p(data['price']), ax = axes[1])
df = data.copy()



# Conveting the string values into numeric values so that our model can interpret it.

for column in df.columns[df.columns.isin(['neighbourhood_group', 'room_type', 'neighbourhood'])]:

        df[column] = df[column].factorize()[0]
df.drop(['id', 'host_id'], axis = 1, inplace = True)
# Plotting the correlation to check the impact of new variables

corr = df.corr(method='spearman')

plt.figure(figsize=(10,8))

sns.heatmap(corr, annot=True)
data.columns
df = df[np.log1p(df['price']) < 7.5]

df = df[np.log1p(df['price']) > 2.5]



# Plotting the new log(Price) curve after removing the outliers

plt.figure(figsize=(5,4))

sns.distplot(np.log1p(df['price']))

plt.show()
df.columns
# Selecting neighbourhood, longitude, room_type, minimum_nights and calculated_host_listing_count on the basis of 

# our analysis of the correlation matrix

x = df.iloc[:,[1,3,4,6,9]]



y = df['price']

y2 = np.log1p(df['price'])
scaler = StandardScaler()

x = scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=7)

x_train1,x_test1,y_train1,y_test1=train_test_split(x,y2,test_size=.1,random_state=7)
model = LinearRegression()

model.fit(x_train1, y_train1)

y_pred1 = model.predict(x_test1)



print('R-Squared ' + str(r2_score(y_test1, y_pred1)))

print('MAE ' + str(mean_absolute_error(y_test1, y_pred1)))

print('MSE ' + str(np.sqrt(mean_squared_error(y_test1, y_pred1))))
model = DecisionTreeRegressor()

model.fit(x_train1, y_train1)

y_pred1 = model.predict(x_test1)



print('R-Squared ' + str(r2_score(y_test1, y_pred1)))

print('MAE ' + str(mean_absolute_error(y_test1, y_pred1)))

print('MSE ' + str(np.sqrt(mean_squared_error(y_test1, y_pred1))))
model = RandomForestRegressor(n_estimators = 100, random_state = 0) 

model.fit(x_train1, y_train1)

y_pred1 = model.predict(x_test1)



print('R-Squared ' + str(r2_score(y_test1, y_pred1)))

print('MAE ' + str(mean_absolute_error(y_test1, y_pred1)))

print('MSE ' + str(np.sqrt(mean_squared_error(y_test1, y_pred1))))
XGB = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 

                             learning_rate=0.05, max_depth=6, 

                             min_child_weight=1.5, n_estimators=7200,

                             reg_alpha=0.9, reg_lambda=0.6,

                             subsample=0.2,seed=42, silent=1)

XGB.fit(x_train1, y_train1)

y_pred1 = XGB.predict(x_test1)



print('R-Squared ' + str(r2_score(y_test1, y_pred1)))

print('MAE ' + str(mean_absolute_error(y_test1, y_pred1)))

print('MSE ' + str(np.sqrt(mean_squared_error(y_test1, y_pred1))))
LightGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

LightGB.fit(x_train1, y_train1)

y_pred1 = LightGB.predict(x_test1)



print('R-Squared ' + str(r2_score(y_test1, y_pred1)))

print('MAE ' + str(mean_absolute_error(y_test1, y_pred1)))

print('MSE ' + str(np.sqrt(mean_squared_error(y_test1, y_pred1))))