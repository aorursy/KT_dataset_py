import numpy as np

import pandas as pd

import pandas_profiling as pp

import seaborn as sns

import matplotlib.pyplot as plt

import os

import datetime
import warnings  

warnings.filterwarnings('ignore')
HOME_PATH = r'../input'

os.listdir(HOME_PATH)
train_set = pd.read_csv(HOME_PATH+'/train.csv')
%pylab inline
def split_datetime(df):

    splitted_date = df['datetime'].str.split('-')

    df['year'] = [int(date[0]) for date in splitted_date]

    df['year'].loc[df['year']==2011] = 1

    df['year'].loc[df['year']==2012] = 2

#     df['month'] = [date[1] for date in splitted_date]

#     df['day'] = [int(date[2].split(" ")[0]) for date in splitted_date]

    df['time'] = [date[2].split(" ")[1] for date in splitted_date]
def round_temp(df):

    df['rounded_temp'] = df['temp'].round()

    df['rounded_atemp'] = df['atemp'].round()
def month_to_num(df):

    """ Convert month to numerical """

    for i in range(1,10):

        df['month'].loc[df['month']=='0'+str(i)] = i

    for i in range(10,13):

        df['month'].loc[df['month']==str(i)] = i
def time_to_num(df):

    """ Convert time to numerical """

    for i in range(0,10):

        df['time'].loc[df['time']=='0'+str(i)+':00:00'] = i

    for i in range(10,24):

        df['time'].loc[df['time']==str(i)+':00:00'] = i
def weekend(df):

    df['weekend'] = np.zeros_like(df['holiday'])

    df['weekend'].loc[(df['workingday'] == 0) & (df['holiday'] == 0)] = 1
def weekday(df):

    df['weekday'] = df['datetime'].apply(lambda date : \

                                         datetime.datetime.strptime(str(date.split()[0]),"%Y-%m-%d").weekday())
def process_df(df):

    split_datetime(df)

    round_temp(df)

#     month_to_num(df)

    time_to_num(df)

    weekend(df)

    weekday(df)

    return df.drop('datetime', axis=1)
pp.ProfileReport(train_set)
train_set = process_df(train_set)
train_set['count_bin'] = np.zeros_like(train_set['count'])

train_set['count_bin'].loc[(train_set['count']>20) & (train_set['count']<100)] = 1

for i in range(2,10):

    train_set['count_bin'].loc[(train_set['count']>(i-1)*100) & (train_set['count']<(i)*100)] = i

train_set['count_bin'].loc[train_set['count']>900] = 10
fig = plt.figure(figsize=(10, 4))

fig.add_subplot(1,2,1)

sns.countplot(x='year', hue='count_bin', data=train_set.loc[train_set['year']==1])

fig.add_subplot(1,2,2)

sns.countplot(x='year', hue='count_bin', data=train_set.loc[train_set['year']==2])
# sns.factorplot(x="month",y="count",data=train_set,kind='bar')
fig = plt.figure(figsize=(20, 6))

fig.add_subplot(1,2,1)

sns.countplot(x='weekday', hue='count_bin', data=train_set)

fig.add_subplot(1,2,2)

sns.countplot(x='time', hue='count_bin', data=train_set)
sns.factorplot(x="weekday",y="count",data=train_set,kind='bar')

sns.factorplot(x="time",y="count",data=train_set,kind='bar')
# train_set['high_time'] = np.zeros_like(train_set['time'])

# train_set['high_time'].loc[(((train_set['time'] > 6) & (train_set['time'] < 15)) | (train_set['time'] == 20))] = 1

# train_set['high_time'].loc[((train_set['time'] == 8) | (train_set['time'] == 16) | (train_set['time'] == 19))] = 2

# train_set['high_time'].loc[((train_set['time'] == 17) | (train_set['time'] == 18))] = 3
# fig = plt.figure(figsize=(10, 4))

# fig.add_subplot(1,2,1)

sns.factorplot(x="weekend",y="count",data=train_set,kind='bar')

# sns.countplot(x='weekend', hue='count_bin', data=train_set.loc[train_set['weekend']==1])

# fig.add_subplot(1,2,2)

sns.factorplot(x="workingday",y="count",data=train_set,kind='bar')

# sns.countplot(x='workingday', hue='count_bin', data=train_set.loc[train_set['workingday']==1])
sns.factorplot(x="rounded_temp",y="count",data=train_set,kind='bar')
sns.factorplot(x="rounded_atemp",y="count",data=train_set,kind='bar')
fig = plt.figure()

fig.add_subplot(2,2,1)

plt.hist(train_set['count'].loc[(train_set['count']<50) & (train_set['season']==1)])



fig.add_subplot(2,2,2)

plt.hist(train_set['count'].loc[(train_set['count']<50) & (train_set['season']==2)])



fig.add_subplot(2,2,3)

plt.hist(train_set['count'].loc[(train_set['count']<50) & (train_set['season']==3)])



fig.add_subplot(2,2,4)

plt.hist(train_set['count'].loc[(train_set['count']<50) & (train_set['season']==4)])
sns.factorplot(x="season",y="count",data=train_set,kind='bar')
sns.factorplot(x="weather",y="count",data=train_set,kind='bar')
train_set['temp_bin'] = np.floor(train_set['temp'])//5
sns.factorplot(x="temp_bin",y="count",data=train_set,kind='bar')
from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import lightgbm as lgb

from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score, make_scorer
train_set = pd.get_dummies(train_set, columns=['season', 'weather', 'weekday', 'holiday'])

# train_set = pd.get_dummies(train_set, columns=['season', 'weather', 'weekday', 'year', 'month', 'time'])
train_set['temp_weather_1'] = train_set['temp'] * train_set['weather_1']

train_set['temp_weather_2'] = train_set['temp'] * train_set['weather_2']

train_set['temp_weather_3'] = train_set['temp'] * train_set['weather_3']

train_set['temp_weather_4'] = train_set['temp'] * train_set['weather_4']
y = train_set.loc[:, 'count']

X = train_set.drop(['count', 'count_bin', 'casual', 'registered'], axis=1) 
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, oob_score=True, min_samples_split=4, max_features=0.9, max_depth=17)

rf.fit(X, y)
scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_log_error')

print("score, root score: ", scores, np.sqrt(np.abs(scores)))
d_importance = pd.DataFrame(columns=['features'], data=X.columns)

d_importance['importance'] = rf.feature_importances_

d_importance.sort_values(by='importance',ascending=False).head(20)
# def RMSLE(y_hat, data):

#     y_true = data.get_label()

#     y_hat = np.round(y_hat)

#     y_hat[y_hat<0]=0

#     return 'rmlse', np.sqrt(mean_squared_log_error(y_true, y_hat)), True
# d_train = lgb.Dataset(X, label=y)

# params = {'objective': 'regression', 'metric': 'rmsle', 'random_state': 501, 'verbose': 0, 'reg_alpha ': 0.1, 'reg_lambda': 0.1}
# lgb_cv = lgb.cv(

#             params, 

#             d_train,

#             metrics = 'rmsle',

#             feval= RMSLE,

#             nfold=5,

#             verbose_eval = 5)
# lgb_model = lgb.train(

#             params, 

#             d_train,

#             feval= RMSLE,

#             verbose_eval = 5)
# d_importance = pd.DataFrame(columns=['features'], data=X.columns)

# d_importance['gain_importance'] = lgb_model.feature_importance(importance_type='gain')

# d_importance['split_importance'] = lgb_model.feature_importance(importance_type='split')

# d_importance.sort_values(by='gain_importance',ascending=False).head(25)
# xgb_model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=7, min_child_weight=4, subsample=0.7, random_state=42)

# xgb_model.fit(X, y)
# def rmsle(y_true, y_hat):

#     y_hat = np.round(y_hat)

#     y_hat[y_hat<0]=0

#     return np.sqrt(mean_squared_log_error(y_true, y_hat))



# rmsle_score = make_scorer(rmsle, greater_is_better=False)
# scores = cross_val_score(xgb_model, X, y, cv=5, scoring=rmsle_score)

# print("scores ", np.abs(scores))
# d_importance = pd.DataFrame(columns=['features'], data=X.columns)

# d_importance['importance'] = xgb_model.feature_importances_

# d_importance.sort_values(by='importance',ascending=False).head(20)
test_set = pd.read_csv(HOME_PATH+'/test.csv')

y_test = test_set['datetime']

test_set = process_df(test_set)

test_set = pd.get_dummies(test_set, columns=['season', 'weather', 'weekday', 'holiday'])
test_set['temp_weather_1'] = test_set['temp'] * test_set['weather_1']

test_set['temp_weather_2'] = test_set['temp'] * test_set['weather_2']

test_set['temp_weather_3'] = test_set['temp'] * test_set['weather_3']

test_set['temp_weather_4'] = test_set['temp'] * test_set['weather_4']



test_set['temp_bin'] = np.floor(test_set['temp'])//5



# test_set['high_time'] = np.zeros_like(test_set['time'])

# test_set['high_time'].loc[(((test_set['time'] > 6) & (test_set['time'] < 15)) | (test_set['time'] == 20))] = 1

# test_set['high_time'].loc[((test_set['time'] == 8) | (test_set['time'] == 16) | (test_set['time'] == 19))] = 2

# test_set['high_time'].loc[((test_set['time'] == 17) | (test_set['time'] == 18))] = 3
test_set.head()
predictions = np.zeros_like(y_test)

predictions = (rf.predict(test_set)).round().astype(int)

predictions[predictions < 0] = 0



submission = pd.concat([y_test, pd.Series(predictions, name="count")], axis=1)

print(submission.head(30))



submission.to_csv("submission.csv", index=False)