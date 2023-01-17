# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df_train.describe()
print(df_train.isnull().sum(axis = 0))
print(df_test.isnull().sum(axis = 0))
fig,ax = plt.subplots(1, 2)
fig.set_size_inches(20,5)
df_train['count'].plot(kind = 'hist', bins=100, ax =ax[0])
df_train['count'].plot(kind = 'box', ax =ax[1])
print('Before removing the outliers ', df_train.shape)
df_train = df_train[abs(df_train['count'] - df_train['count'].mean()) < 3*df_train['count'].std()]
print('After removing the outliers ', df_train.shape)
df_train.reset_index(drop = True, inplace = True)
df_train.head()
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(20, 10)
sns.distplot(df_train["count"], ax = ax[0])
sns.distplot(df_train["casual"], ax = ax[1])
sns.distplot(df_train["registered"], ax = ax[2])
df_train['count'] = np.log(df_train['count'] + 1)
df_train['registered'] = np.log(df_train['registered'] + 1)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(20, 10)
sns.distplot(df_train["count"], ax = ax[0])
sns.distplot(df_train["casual"], ax = ax[1])
sns.distplot(df_train["registered"], ax = ax[2])

def add_month(dataframe):
    month = pd.DatetimeIndex(dataframe['datetime']).month
    return month

def add_time(dataframe):
    time = pd.DatetimeIndex(dataframe['datetime']).hour
    return time

def add_year(dataframe):
    year = pd.DatetimeIndex(dataframe['datetime']).year
    return year

def add_day(dataframe):
    day = pd.DatetimeIndex(dataframe['datetime']).dayofweek
    return day
df_train['month'] = add_month(df_train)
df_train['time'] = add_time(df_train)
df_train['year'] = add_year(df_train)
df_train['day'] = add_day(df_train)
df_train.head()
corr = df_train.corr()
corr
df_train.groupby('time')['count'].mean().plot(kind = 'bar')
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20,5)
df_train.groupby('month')['count'].mean().plot('bar', ax = ax[0])
df_train.groupby('season')['count'].mean().plot('bar', ax = ax[1])
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20,5)
df_train.groupby('temp')['count'].mean().plot('bar', ax = ax[0])
df_train.groupby('humidity')['count'].mean().plot('bar', ax = ax[1])
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20,5)
df_train.groupby('holiday')['count'].mean().plot('bar', ax = ax[0])
df_train.groupby('workingday')['count'].mean().plot('bar', ax = ax[1])
print("mean of count according to holidays ", df_train.groupby('holiday')['count'].mean())
print("No of holiday = 1 and holdays = 0 ", df_train.groupby('holiday')['count'].count())
df_train[df_train['windspeed'] == 0].shape
df_train.groupby('windspeed')['count'].count().plot(kind='bar')
df_train.groupby('windspeed')['count'].count()
df_train_windspeed_0 = df_train[df_train['windspeed'] == 0]
df_train_windspeed_not_0 = df_train[df_train['windspeed'] != 0]
print(df_train_windspeed_0.head())
print(df_train_windspeed_not_0.head())
print(df_train_windspeed_0.shape)
print(df_train_windspeed_not_0.shape)
columns_for_windspeed = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day', 'temp', 'humidity']
from sklearn.ensemble import RandomForestRegressor
rf_windspeed = RandomForestRegressor().fit(df_train_windspeed_not_0[columns_for_windspeed], df_train_windspeed_not_0['windspeed'])
df_train_windspeed_0['windspeed'] = rf_windspeed.predict(df_train_windspeed_0[columns_for_windspeed])

df_train = df_train_windspeed_0.append(df_train_windspeed_not_0, sort = 'datetime')
print(df_train.shape)
df_train.head()
print(df_train[df_train['windspeed'] == 0])
df_train.groupby('windspeed')['count'].count()
categorical_columns = ['holiday', 'season', 'workingday', 'weather', 'month', 'time', 'year', 'day']
for category in categorical_columns:
    df_train = df_train.join(pd.get_dummies(df_train[category], prefix = category))
    
df_train.head()
'''
def one_hot_encode(dataframe, column):
    for i in dataframe.groupby(column).count().index:
        s = column + "_" + str(i)
        a = []
        for element in dataframe[column]:
            if element == i:
                a.append(1)
            else:
                a.append(0)
        dataframe[s] = a
    return dataframe
'''
df_train.columns
def normalize(dataframe, columns):
    for column in columns:
        dataframe[column]=((dataframe[column]-dataframe[column].min())/(dataframe[column].max()-dataframe[column].min()))
    return dataframe
df_train = normalize(df_train, columns=['temp', 'humidity', 'windspeed'])
df_train.head()
def remove_columns(dataframe, columns):
    dataframe = dataframe.drop(columns, axis = 1)
    return dataframe
df_train = remove_columns(df_train, ['datetime', 'atemp']) 
print(df_train.columns)
df_train.head()
df_train_y = df_train[['count', 'casual', 'registered']]
df_train_x = remove_columns(df_train, ['casual', 'registered', 'count'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.15, random_state=42)
y_train_casual = y_train['casual']
y_train_registered = y_train['registered']
y_train_total = y_train['count']
y_test_casual = y_test['casual']
y_test_registered = y_test['registered']
y_test_total = y_test['count']
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
all_predictions = []

#as we have the one hot vector we will remove this categorical data
categorical_data = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day']
lr_train_x = remove_columns(x_train, categorical_data)
lr_test_x = remove_columns(x_test, categorical_data)

lr = LinearRegression().fit(lr_train_x, y_train_total)
lr_predictions_on_test_data = np.exp(lr.predict(lr_test_x)) - 1

lr_predictions_on_train_data = np.exp(lr.predict(lr_train_x))

all_predictions.append(lr_predictions_on_train_data)
all_predictions.append(lr_predictions_on_test_data)

for i, prediction in enumerate(all_predictions):
    pre = []
    for p in prediction:
        if p < 0:
            pre.append(0)
        else:
            pre.append(p)
    if i == 0:
        print(np.sqrt(mean_squared_log_error( np.exp(y_train_total)-1, pre )))
    else:
        print(np.sqrt(mean_squared_log_error( np.exp(y_test_total)-1, pre )))
all_predictions = []

training_columns = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day', 'temp', 'humidity', 'windspeed']
train_x = x_train[training_columns]
test_x = x_test[training_columns]

rf = RandomForestRegressor(n_estimators=100, max_depth = 10, min_samples_split=5).fit(train_x, y_train_total)
predictions_on_test_data = np.exp(rf.predict(test_x)) - 1

predictions_on_train_data = np.exp(rf.predict(train_x))

all_predictions.append(predictions_on_train_data)
all_predictions.append(predictions_on_test_data)

for i, prediction in enumerate(all_predictions):
    pre = []
    for p in prediction:
        if p < 0:
            pre.append(0)
        else:
            pre.append(p)
    if i == 0:
        print(np.sqrt(mean_squared_log_error( np.exp(y_train_total)-1, pre )))
    else:
        print(np.sqrt(mean_squared_log_error( np.exp(y_test_total)-1, pre )))
all_predictions = []

training_columns = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day', 'temp', 'humidity', 'windspeed']
train_x = x_train[training_columns]
test_x = x_test[training_columns]

rf_casual = RandomForestRegressor(n_estimators=300, max_depth = 10, min_samples_split=8).fit(train_x, y_train_casual)
predictions_casual = rf_casual.predict(test_x)


rf_registered = RandomForestRegressor().fit(train_x, y_train_registered)
predictions_registered = np.exp(rf_registered.predict(test_x))-1

predictions = predictions_casual + predictions_registered

predictions_casual_train = rf_casual.predict(train_x)
predictions_registered_train = np.exp(rf_registered.predict(train_x))-1

predictions_train = predictions_casual_train + predictions_registered_train

all_predictions.append(predictions_train)
all_predictions.append(predictions)

for i, prediction in enumerate(all_predictions):
    pre = []
    for p in prediction:
        if p < 0:
            pre.append(0)
        else:
            pre.append(p)
    if i == 0:
        print(np.sqrt(mean_squared_log_error( np.exp(y_train_total)-1, pre )))
    else:
        print(np.sqrt(mean_squared_log_error( np.exp(y_test_total)-1, pre )))
df_test.head()
df_test['month'] = add_month(df_test)
df_test['time'] = add_time(df_test)
df_test['year'] = add_year(df_test)
df_test['day'] = add_day(df_test)
df_test.head()
df_test_windspeed_0 = df_test[df_test['windspeed'] == 0]
df_test_windspeed_not_0 = df_test[df_test['windspeed'] != 0]
columns_for_windspeed = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day', 'temp', 'humidity']

df_test_windspeed_0['windspeed'] = rf_windspeed.predict(df_test_windspeed_0[columns_for_windspeed])

df_test = df_test_windspeed_0.append(df_test_windspeed_not_0, sort = 'datetime')
df_test.head()
df_test = df_test.sort_values(by='datetime')
df_test.head()
categorical_columns = ['holiday', 'season', 'workingday', 'weather', 'month', 'time', 'year', 'day']
for category in categorical_columns:
    df_test = df_test.join(pd.get_dummies(df_test[category], prefix = category))
    
print(df_test.head())
print(df_test.columns)
df_test = normalize(df_test, columns=['temp', 'humidity', 'windspeed'])
df_test.head()
df_datetime = df_test['datetime']
df_test = remove_columns(df_test, ['datetime', 'atemp']) 
print(df_test.columns)
df_test.columns.shape == df_train_x.columns.shape
training_columns = ['holiday', 'season', 'workingday', 'month', 'time', 'year', 'day', 'temp', 'humidity', 'windspeed']
df_test_final = df_test[training_columns]
predictions = np.exp(rf.predict(df_test_final))-1
'''
predictions_casual = rf_casual.predict(df_test_final)

predictions_registered = np.exp(rf_registered.predict(df_test_final))-1

predictions = predictions_casual + predictions_registered

print(predictions[:5])
'''

data = {'datetime': df_datetime, 'count': predictions}
df = pd.DataFrame(data)
df.head()
df.to_csv('submission.csv', index = False)