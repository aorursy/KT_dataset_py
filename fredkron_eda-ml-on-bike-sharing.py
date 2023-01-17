import calendar
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime
from scipy import stats

%matplotlib inline
sns.set()
df = pd.read_csv("../input/train.csv")
df.head()
drop_lst = ['casual', 'registered']
df = df.drop(drop_lst, axis=1)
df.head()
df.info()
df.describe()
df['count'].head()
df['count'].describe()
plt.hist(df['count']);
count_log = np.log(df['count'])
plt.hist(count_log);
count_boxcox, _ = stats.boxcox(df['count'])
count_boxcox
plt.hist(count_boxcox);
df['count_log'] = count_log
df['count_boxcox'] = count_boxcox
df.head()
df['datetime'] = pd.to_datetime(df['datetime'])
df.head()
df.info()
df['dow'] = df['datetime'].dt.dayofweek
df.head()
df['month'] = df['datetime'].dt.month
df.head()
df['week'] = df['datetime'].dt.week
df.head()
df['hour'] = df['datetime'].dt.hour
df.head()
df['year'] = df['datetime'].dt.year
df.head()
df['day'] = df['datetime'].dt.day
df.head()
df = df.set_index(df['datetime'])
df.head()
df = df.drop(labels='datetime', axis=1)
df.head()
df['season'].describe()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['1', '2', '3', '4']

values = df['season'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['season'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Seasons in 2011 & 2012');
spring_2011 = int(df['season'][df['season'] == 1][df['year'] == 2011].value_counts())
summer_2011 = int(df['season'][df['season'] == 2][df['year'] == 2011].value_counts())
fall_2011 = int(df['season'][df['season'] == 3][df['year'] == 2011].value_counts())
winter_2011 = int(df['season'][df['season'] == 4][df['year'] == 2011].value_counts())

spring_2012 = int(df['season'][df['season'] == 1][df['year'] == 2012].value_counts())
summer_2012 = int(df['season'][df['season'] == 2][df['year'] == 2012].value_counts())
fall_2012 = int(df['season'][df['season'] == 3][df['year'] == 2012].value_counts())
winter_2012 =int(df['season'][df['season'] == 4][df['year'] == 2012].value_counts())

print("Spring 2011: {}".format(spring_2011))
print("Summer 2011: {}".format(summer_2011))
print("Fall 2011: {}".format(fall_2011))
print("Winter 2011: {}".format(winter_2011))
print("-----------------------------------------")
print("Spring 2012: {}".format(spring_2012))
print("Summer 2012: {}".format(summer_2012))
print("Fall 2012: {}".format(fall_2012))
print("Winter 2012: {}".format(winter_2012))
df['holiday'].describe()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['0', '1']

values = df['holiday'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['holiday'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Holidays in 2011 & 2012');
no_holiday_2011 = int(df['holiday'][df['holiday'] == 0][df['year'] == 2011].value_counts())
holiday_2011 = int(df['holiday'][df['holiday'] == 1][df['year'] == 2011].value_counts())
no_holiday_2012 = int(df['holiday'][df['holiday'] == 0][df['year'] == 2012].value_counts())
holiday_2012 = int(df['holiday'][df['holiday'] == 1][df['year'] == 2012].value_counts())

print("No Holidays 2011: {}".format(no_holiday_2011))
print("No Holidays 2012: {}".format(no_holiday_2012))
print("Holidays 2011: {}".format(holiday_2011))
print("Holidays 2012: {}".format(holiday_2012))
print('----------------')
total_2011 = no_holiday_2011 + holiday_2011
total_2012 = no_holiday_2012 + holiday_2012
print('No Holidays 2011: {:.0f}%'.format(no_holiday_2011 / total_2011 * 100))
print('No Holidays 2012: {:.0f}%'.format(no_holiday_2012 / total_2012 * 100))
df['workingday'].describe()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names = ['0', '1']

values = df['workingday'][df['year'] == 2011].value_counts()
ax[0].bar(names, values)

values = df['workingday'][df['year'] == 2012].value_counts()
ax[1].bar(names, values)

fig.suptitle('Working day in 2011 & 2012');
no_workingday_2011 = int(df['workingday'][df['workingday'] == 0][df['year'] == 2011].value_counts())
workingday_2011 = int(df['workingday'][df['workingday'] == 1][df['year'] == 2011].value_counts())
no_workingday_2012 = int(df['workingday'][df['workingday'] == 0][df['year'] == 2012].value_counts())
workingday_2012 = int(df['workingday'][df['workingday'] == 1][df['year'] == 2012].value_counts())

print("No working day 2011: {}".format(no_workingday_2011))
print("working day 2011: {}".format(workingday_2011))
print("No working day 2012: {}".format(no_workingday_2012))
print("working day 2012: {}".format(workingday_2012))
print('----------------')
total_2011 = no_workingday_2011 + workingday_2011
total_2012 = no_workingday_2012 + workingday_2012
print('No working day 2011: {:.0f}%'.format(no_workingday_2011 / total_2011 * 100))
print('No working day 2012: {:.0f}%'.format(no_workingday_2012 / total_2012 * 100))
df['weather'].describe()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

names_2011 = ['1', '2', '3']
names_2012 = ['1', '2', '3', '4']

values = df['weather'][df['year'] == 2011].value_counts()
ax[0].bar(names_2011, values)

values = df['weather'][df['year'] == 2012].value_counts()
ax[1].bar(names_2012, values)

fig.suptitle('Weather in 2011 & 2012');
weather_2011_1 = df['weather'][df['weather'] == 1][df['year'] == 2011].value_counts()
weather_2011_2 = df['weather'][df['weather'] == 2][df['year'] == 2011].value_counts()
weather_2011_3 = df['weather'][df['weather'] == 3][df['year'] == 2011].value_counts()

weather_2012_1 = df['weather'][df['weather'] == 1][df['year'] == 2012].value_counts()
weather_2012_2 = df['weather'][df['weather'] == 2][df['year'] == 2012].value_counts()
weather_2012_3 = df['weather'][df['weather'] == 3][df['year'] == 2012].value_counts()
weather_2012_4 = df['weather'][df['weather'] == 4][df['year'] == 2012].value_counts()

print('weather_1 in 2011: {}'.format(int(weather_2011_1)))
print('weather_2 in 2011: {}'.format(int(weather_2011_2)))
print('weather_3 in 2011: {}'.format(int(weather_2011_3)))
print('--------------')
print('weather_1 in 2012: {}'.format(int(weather_2012_1)))
print('weather_2 in 2012: {}'.format(int(weather_2012_2)))
print('weather_3 in 2012: {}'.format(int(weather_2012_3)))
print('weather_4 in 2012: {}'.format(int(weather_2012_4)))
print('---------------')
total_2011 = int(weather_2011_1) + int(weather_2011_2) + int(weather_2011_3)
total_2012 = int(weather_2012_1) + int(weather_2012_2) + int(weather_2012_3) + int(weather_2012_4)
print('weather_1 in 2011: {:.0f}%'.format(int(weather_2011_1) / int(total_2011) * 100))
print('weather_2 in 2011: {:.0f}%'.format(int(weather_2011_2) / int(total_2011) * 100))
print('weather_3 in 2011: {:.0f}%'.format(int(weather_2011_3) / int(total_2011) * 100))
print('--------------')
print('weather_1 in 2012: {:.0f}%'.format(int(weather_2012_1) / int(total_2012) * 100))
print('weather_2 in 2012: {:.0f}%'.format(int(weather_2012_2) / int(total_2012) * 100))
print('weather_3 in 2012: {:.0f}%'.format(int(weather_2012_3) / int(total_2012) * 100))
print('weather_4 in 2012: {:.0f}%'.format(int(weather_2012_4) / int(total_2012) * 100))
df['temp'].describe()
plt.hist(df['temp'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['temp'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');
df['atemp'].describe()
plt.hist(df['atemp'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['atemp'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');
df['humidity'].describe()
plt.hist(df['humidity'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['humidity'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');
df['windspeed'].describe()
plt.hist(df['windspeed'][df['year'] == 2011], alpha=0.5, label='2011')
plt.hist(df['windspeed'][df['year'] == 2012], alpha=0.5, label='2012')

plt.legend(loc='upper right');
df['dow'].describe()
plt.hist(df['dow'][df['year'] == 2011], alpha=0.5, label='2011', bins=7)
plt.hist(df['dow'][df['year'] == 2012], alpha=0.5, label='2012', bins=7)

plt.legend(loc='upper right');
df['month'].describe()
plt.hist(df['month'][df['year'] == 2011], alpha=0.5, label='2011', bins=12)
plt.hist(df['month'][df['year'] == 2012], alpha=0.5, label='2012', bins=12)

plt.legend(loc='upper right');
df['week'].describe()
plt.hist(df['week'][df['year'] == 2011], alpha=0.5, label='2011', bins=52)
plt.hist(df['week'][df['year'] == 2012], alpha=0.5, label='2012', bins=52)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend(loc='upper right');
df['hour'].describe()
plt.hist(df['hour'][df['year'] == 2011], alpha=0.5, label='2011', bins=24)
plt.hist(df['hour'][df['year'] == 2012], alpha=0.5, label='2012', bins=24)
plt.legend(loc='upper right');
df['day'].describe()
plt.hist(df['day'][df['year'] == 2011], alpha=0.5, label='2011', bins=31)
plt.hist(df['day'][df['year'] == 2012], alpha=0.5, label='2012', bins=31)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend(loc='upper right');
df['year'].describe()
names = ['2011', '2012']
values = df['year'].value_counts()
plt.bar(names, values);
count_2011 = df['year'][df['year'] == 2011].count()
count_2012 = df['year'][df['year'] == 2012].count()

print('2011: {}'.format(count_2011))
print('2012: {}'.format(count_2012))
cor_mat = df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig = plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True);
sns.pointplot(x=df['temp'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);
from scipy import stats
_, _, r_value, _, _ = stats.linregress(df['count'], df['temp'])
r_square = r_value ** 2
r_square.round(2)
sns.pointplot(x=df['atemp'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);
_, _, r_value, _, _ = stats.linregress(df['count'], df['atemp'])
r_square = r_value ** 2
r_square.round(2)
sns.pointplot(x=df['hour'], y=df['count'])
fig = plt.gcf()
fig.set_size_inches(30,12);
sns.pointplot(x=df['temp'], y=df['atemp'])
fig = plt.gcf()
fig.set_size_inches(30,12);
_, _, r_value, _, _ = stats.linregress(df['temp'], df['atemp'])
r_square = r_value ** 2
r_square.round(2)
df = df.drop(labels='atemp', axis=1)
df.head()
df = df.drop(labels='count_log', axis=1)
df = df.drop(labels='count_boxcox', axis=1)
df.head()
df = pd.get_dummies(df, columns=['weather'])
df.head()
df = df.drop(labels='weather_4', axis=1)
df.head()
df['temp_weath_1'] = df['temp'] * df['weather_1']
df['temp_weath_2'] = df['temp'] * df['weather_2']
df['temp_weath_3'] = df['temp'] * df['weather_3']
df['temp_weath_1'] = df['temp_weath_1'].astype(int)
df['temp_weath_2'] = df['temp_weath_2'].astype(int)
df['temp_weath_3'] = df['temp_weath_3'].astype(int)
df.head()
X = df.loc[:, df.columns != 'count']
y = np.log(df['count'])
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, QuantileTransformer, RobustScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBRegressor
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('LASSO', Lasso(random_state=42))])))
pipelines.append(('ScaledRID', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('RID', Ridge(random_state=42))])))
pipelines.append(('ScaledKNN', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor(n_neighbors=2))])))
pipelines.append(('ScaledCART', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor(random_state=42))])))
pipelines.append(('ScaledGBM', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor(random_state=42))])))
pipelines.append(('ScaledRFR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('RFR', RandomForestRegressor(random_state=42))])))
pipelines.append(('ScaledSVR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('SVR', SVR(kernel='linear'))])))
pipelines.append(('ScaledXGBR', Pipeline([('poly', PolynomialFeatures()), ('Scaler', StandardScaler()), ('XGBR', XGBRegressor(random_state=42))])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(random_state=42)
    cv_results = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_log_error')
    results.append(np.sqrt(cv_results))
    names.append(name)
    msg = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
    print(msg)
df_test = pd.read_csv("../input/test.csv")
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test['dow'] = df_test['datetime'].dt.dayofweek
df_test['month'] = df_test['datetime'].dt.month
df_test['week'] = df_test['datetime'].dt.week
df_test['hour'] = df_test['datetime'].dt.hour
df_test['year'] = df_test['datetime'].dt.year
df_test['day'] = df_test['datetime'].dt.day
df_test = df_test.set_index(df_test['datetime'])
df_test = df_test.drop(labels='datetime', axis=1)
df_test = df_test.drop(labels='atemp', axis=1)
df_test = pd.get_dummies(df_test, columns=['weather'])
df_test = df_test.drop(labels='weather_4', axis=1)
df_test['temp_weath_1'] = df_test['temp'] * df_test['weather_1']
df_test['temp_weath_2'] = df_test['temp'] * df_test['weather_2']
df_test['temp_weath_3'] = df_test['temp'] * df_test['weather_3']
df_test['temp_weath_1'] = df_test['temp_weath_1'].astype(int)
df_test['temp_weath_2'] = df_test['temp_weath_2'].astype(int)
df_test['temp_weath_3'] = df_test['temp_weath_3'].astype(int)
standardscaler = StandardScaler()
model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=7, min_child_weight=4, subsample=0.7, random_state=42)
model.fit(X_train, y_train)
model.predict(df_test)
pipe = Pipeline([('poly', PolynomialFeatures()), ('StandardScaler', standardscaler), ('XGBR', model)])
pipe.fit(X_train, y_train)
y_pred = np.exp(pipe.predict(df_test))
y_pred
df_test['count'] = y_pred
df_test.head()
df_test[['count']].to_csv('submission.csv', index=True)
df_test[['count']].head()
