%matplotlib inline

import math

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



# Our Models

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

solution = pd.read_csv('../input/sample_sol.csv')
print('Dimensions:', train.shape, '\n')

print('Types:', train.dtypes, '\n')

print('Missing Values:', train.isnull().sum(), '\n')



# Get 5 random samples

train.sample(n = 5)
fig, axes = plt.subplots(len(train.columns)//3, 3, figsize=(10, 10))



i = 0

for triaxis in axes:

    for axis in triaxis:

        train.hist(column = train.columns[i], ax=axis)

        i = i + 1
sns.distplot(train["cnt"], color ='blue')
plt.figure(figsize=(15, 15))



plt.subplot(3, 4, 1)

train.groupby('mnth').cnt.sum().plot()



plt.subplot(3, 4, 2)

train.groupby('hr').cnt.sum().plot()



plt.subplot(3, 4, 3)

train.groupby('temp').cnt.sum().plot()



plt.subplot(3, 4, 4)

train.groupby('atemp').cnt.sum().plot()



plt.subplot(3, 4, 5)

train.groupby('season').cnt.sum().plot()



plt.subplot(3, 4, 6)

train.groupby('weathersit').cnt.sum().plot()



plt.subplot(3, 4, 7)

train.groupby('hum').cnt.sum().plot()



plt.subplot(3, 4, 8)

train.groupby('windspeed').cnt.sum().plot()



plt.show()
plt.figure(figsize=(10, 10))

sns.swarmplot(x = 'hr', y = 'temp', data = train, hue = 'season')

plt.show()
plt.figure(figsize=(15, 15))



plt.subplot(3, 3, 1)

plt.xlabel('temp')

plt.ylabel('atemp')

plt.scatter(x = train.temp, y = train.atemp)





plt.subplot(3, 3, 2)

plt.xlabel('windspeed')

plt.ylabel('temp')

plt.scatter(x = train.windspeed, y = train.temp)



plt.subplot(3, 3, 3)

plt.xlabel('weathersit')

plt.ylabel('temp')

plt.scatter(x = train.weathersit, y = train.temp)



plt.subplot(3, 3, 4)

plt.xlabel('hum')

plt.ylabel('temp')

plt.scatter(x = train.hum, y = train.temp)



plt.subplot(3, 3, 5)

plt.xlabel('season')

plt.ylabel('mnth')

plt.scatter(x = train.season, y = train.mnth)



plt.show()
plt.figure(figsize=(10, 10))



plt.subplot(1, 2, 1)



workdays = train.loc[train.workingday == 1]

workdays.groupby('hr').cnt.sum().plot()



plt.subplot(1, 2, 2)

not_workdays = train.loc[train.workingday == 0]

not_workdays.groupby('hr').cnt.sum().plot()
train.loc[train.mnth == 12, 'season'] = 4

test.loc[test.mnth == 12, 'season'] = 4



plt.xlabel('season')

plt.ylabel('mnth')

plt.scatter(x = train.season, y = train.mnth)



plt.show()
season_dummies = pd.get_dummies(train.season, prefix = 'season')

train = pd.concat([train, season_dummies], axis = 1)



season_dummies = pd.get_dummies(test.season, prefix = 'season')

test = pd.concat([test, season_dummies], axis = 1)



weathersit_dummies = pd.get_dummies(train.weathersit, prefix = 'weathersit')

train = pd.concat([train, weathersit_dummies], axis = 1)



weathersit_dummies = pd.get_dummies(test.weathersit, prefix = 'weathersit')

test = pd.concat([test, weathersit_dummies], axis = 1)



train.head()
train = train.drop(['season', 'weathersit'], axis = 1)

test = test.drop(['season', 'weathersit'], axis = 1)
train = train.drop(['atemp'], axis = 1)

test = test.drop(['atemp'], axis = 1)
plt.figure(figsize=(10, 10))



sns.heatmap(train.corr())

plt.show
train['hour_best'] = train[['hr', 'workingday']].apply(lambda is_best: (0, 1)

                                                      [((is_best['workingday'] == 1 and  (7 <= is_best['hr'] <= 9 or 17 <= is_best['hr'] <= 18)) or

                                                       (is_best['workingday'] == 0 and  (11 <= is_best['hr'] <= 18)))], axis = 1)



test['hour_best'] = test[['hr', 'workingday']].apply(lambda is_best: (0, 1)

                                                      [((is_best['workingday'] == 1 and  (7 <= is_best['hr'] <= 9 or 17 <= is_best['hr'] <= 18)) or

                                                       (is_best['workingday'] == 0 and  (11 <= is_best['hr'] <= 18)))], axis = 1)
train.sample(n = 5)
train_set, test_set = train_test_split(train, test_size = 0.3, random_state = 42)



x_train = train_set.drop('cnt', axis = 1)

y_train = train_set.cnt



x_test = test_set.drop('cnt', axis = 1)

y_test = test_set.cnt
et_model = ExtraTreesRegressor(n_estimators = 800, random_state = 42, n_jobs = -1, max_features = 9, min_samples_split = 4, max_depth = 20)

et_model.fit(x_train, y_train)



et_y_pred = np.floor(et_model.predict(x_test))



print(np.sqrt(mean_squared_error(y_test, et_y_pred)))

print(et_model.feature_importances_)



plt.plot([0,1000],[0,1000], color='red')

plt.scatter(et_y_pred, y_test)

plt.show()
rf_model = RandomForestRegressor(n_estimators = 750, random_state = 42, n_jobs = -1, max_features = 8, min_samples_split = 2)

rf_model.fit(x_train, y_train)



rf_y_pred = np.floor(rf_model.predict(x_test))



print(np.sqrt(mean_squared_error(y_test, rf_y_pred)))

print(rf_model.feature_importances_)



plt.plot([0,1000],[0,1000], color='red')

plt.scatter(rf_y_pred, y_test)

plt.show()
xgb_model = XGBRegressor(n_estimators = 4500, early_stopping_rounds=10, learning_rate = 0.12,

                     random_state = 42, nthread = 64, max_depth = 10, gamma = 4, reg_lambda = 5,

                     reg_alpha = 13)

xgb_model.fit(x_train, y_train)



xgb_y_pred = np.floor(xgb_model.predict(x_test))



print(np.sqrt(mean_squared_error(y_test, xgb_y_pred)))

print(xgb_model.feature_importances_)



plt.plot([0,1000],[0,1000], color='red')

plt.scatter(xgb_y_pred, y_test)

plt.show()
cat_model = CatBoostRegressor(task_type = 'GPU', iterations = 2000, verbose = False)

cat_model.fit(x_train, y_train)



cat_y_pred = np.floor(cat_model.predict(x_test))



print(np.sqrt(mean_squared_error(y_test, cat_y_pred)))

print(cat_model.feature_importances_)



plt.plot([0,1000],[0,1000], color='red')

plt.scatter(cat_y_pred, y_test)

plt.show()
def averagingModels(X, train, labels, models=[]):

    for model in models:

        model.fit(train, labels)

    predictions = np.column_stack([

        model.predict(X) for model in models

    ])

    return np.mean(predictions, axis=1)
y_pred = averagingModels(x_test, x_train, y_train, [et_model, rf_model, xgb_model, cat_model])



print('Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))



plt.plot([0,1000],[0,1000], color='red')

plt.scatter(xgb_y_pred, y_test)

plt.show()
full_set = train.copy()



x_full = full_set.drop(['cnt'], axis = 1)

y_full = full_set.cnt



y_solution_predict = averagingModels(test, x_full, y_full, [et_model, rf_model, xgb_model, cat_model])



solution['cnt'] = np.floor(y_solution_predict).clip(0)



solution['cnt'] = solution['cnt'].astype('int')



solution.to_csv('solution.csv', index = False)