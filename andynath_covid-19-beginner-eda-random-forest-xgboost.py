import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys

np.set_printoptions(threshold=sys.maxsize)
path_train = 'covid19-global-forecasting-week-1/train-2.csv'

path_test = 'covid19-global-forecasting-week-1/test.csv'

path_sbumit = 'covid19-global-forecasting-week-1/submission.csv'



train_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/train.csv'

test_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/test.csv'

submit_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/submission.csv'



df_train = pd.read_csv(train_kaggle)

df_test = pd.read_csv(test_kaggle)
df_train.info()
df_test.info()
# Dataset Dimesnions

print('Train shape', df_train.shape)

print('Test shape', df_test.shape)

# Missing/Null Values

print('\nTrain Missing\n', df_train.isnull().sum())

print('\nTest Missing\n', df_test.isnull().sum())
lst = df_train['Country/Region'].unique()

print('Total_Countries\n:', len(lst))

for i in lst:

    print(i)
print(df_train['Date'].min(), ' - ', df_train['Date'].max())
# GroupBy syntax (columns to group by in list)[Columns to aggregate, apply function to] . aggregation functions on it 

train_cases_conf = df_train.groupby(['Date'])['ConfirmedCases'].sum()

train_cases_conf
train_cases_conf.plot(figsize = (10,8), title = 'Worldwide Confirmed Cases')
train_fatal = df_train.groupby(['Date'])['Fatalities'].sum()

train_fatal
train_fatal.plot(figsize = (10,8), title = 'Worldwide Fatalaties')
def country_stats(country, df):

    country_filt = (df['Country/Region'] == country)

    df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()

    df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()

    fig, axes = plt.subplots(nrows = 2, ncols= 1, figsize=(15,15))

    df_cases.plot(ax = axes[0])

    df_fatal.plot(ax = axes[1])

    

country_stats('US', df_train)
# grouping using same Country filter to get fatalities on each date (grouped by date)

# groupby([list of columns to groupby]) [which columns to apply aggregate fx to ]. (aggregate function)

# To Do - Fix Ticks 



def country_stats_log(country, df):

    count_filt =(df_train['Country/Region'] == country)

    df_count_case = df_train.loc[count_filt].groupby(['Date'])['ConfirmedCases'].sum()

    df_count_fatal = df_train.loc[count_filt].groupby(['Date'])['Fatalities'].sum()

    plt.figure(figsize=(15,10))

    plt.axes(yscale = 'log')

    plt.plot(df_count_case.index, df_count_case.tolist(), 'b', label = country +' Total Confirmed Cases')

    plt.plot(df_count_fatal.index, df_count_fatal.tolist(), 'r', label = country +' Total Fatalities')

    plt.title(country +' COVID Cases and Fatalities (Log Scale)')

    plt.legend()

    



country_stats_log('US', df_train)
# as_index = False to not make the grouping column the index, creates a df here instead of series, preserves

# Confirmedcases column



train_case_country = df_train.groupby(['Country/Region'], as_index=False)['ConfirmedCases'].max()



# Sorting by number of cases

train_case_country.sort_values('ConfirmedCases', ascending=False, inplace = True)

train_case_country
plt.figure(figsize=(8,6))

plt.bar(train_case_country['Country/Region'][:5], train_case_country['ConfirmedCases'][:5], color = ['red', 'yellow','black','blue','green'])
# Confirmed Cases till a particular day by country



def case_day_country (Date, df):

    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()

    date_filter = (df['Date'] == Date)

    df = df.loc[date_filter]

    df.sort_values('ConfirmedCases', ascending = False, inplace = True)

    sns.catplot(x = 'Country/Region', y = 'ConfirmedCases' , data = df.head(10), height=5,aspect=3, kind = 'bar')

    

    

case_day_country('2020-03-23', df_train)
df_train.Date = pd.to_datetime(df_train['Date'])

print(df_train['Date'].max())

print(df_test['Date'].min())
date_filter = df_train['Date'] < df_test['Date'].min()

df_train = df_train.loc[date_filter]
# Dropping ID and getting rid of Province/State with NULL values 

df_train.info()
# lets get Cumulative sum of ConfirmedCases and Fatalities for each country on each data (same as original data)

# Doing to create copy without ID and 



train_country_date = df_train.groupby(['Country/Region', 'Date', 'Lat', 'Long'], as_index=False)['ConfirmedCases', 'Fatalities'].sum()
print(train_country_date.info())

print(train_country_date.isnull().sum())
train_country_date.info()
# Adding day, month, day of week columns 



train_country_date['Month'] = train_country_date['Date'].dt.month

train_country_date['Day'] = train_country_date['Date'].dt.day

train_country_date['Day_Week'] = train_country_date['Date'].dt.dayofweek

train_country_date['quarter'] = train_country_date['Date'].dt.quarter

train_country_date['dayofyear'] = train_country_date['Date'].dt.dayofyear

train_country_date['weekofyear'] = train_country_date['Date'].dt.weekofyear
train_country_date.head()
train_country_date.info()
# First drop Province/State

df_test.drop('Province/State', axis = 1, inplace = True)



# Converting Date Object to Datetime type



df_test.Date = pd.to_datetime(df_test['Date'])

df_test.head(2)
# adding Month, DAy, Day_week columns Using Pandas Series.dt.month



df_test['Month'] = df_test['Date'].dt.month

df_test['Day'] = df_test['Date'].dt.day

df_test['Day_Week'] = df_test['Date'].dt.dayofweek

df_test['quarter'] = df_test['Date'].dt.quarter

df_test['dayofyear'] = df_test['Date'].dt.dayofyear

df_test['weekofyear'] = df_test['Date'].dt.weekofyear
df_test.info()
# train_country_date

# df_test

# Lets select the Common Labels and concatenate.



labels = ['Country/Region', 'Lat', 'Long', 'Date', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear']



df_train_clean = train_country_date[labels]

df_test_clean = df_test[labels]



data_clean = pd.concat([df_train_clean, df_test_clean], axis = 0)
data_clean.info()
from sklearn.preprocessing import LabelEncoder
# Label Encoder for Countries 



enc = LabelEncoder()

data_clean['Country'] = enc.fit_transform(data_clean['Country/Region'])

data_clean
# Dropping Country/Region and Date



data_clean.drop(['Country/Region', 'Date'], axis = 1, inplace=True)
index_split = df_train.shape[0]

data_train_clean = data_clean[:index_split]
data_test_clean = data_clean[index_split:]
data_train_clean.tail(5)
x = data_train_clean[['Lat', 'Long', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear', 'Country']]

y_case = df_train['ConfirmedCases']

y_fatal = df_train['Fatalities']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y_case, test_size = 0.3, random_state = 42)
from sklearn.model_selection import train_test_split



x_train_fatal, x_test_fatal, y_train_fatal, y_test_fatal = train_test_split(x, y_fatal, test_size = 0.3, random_state = 42)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor(n_estimators =100)

rf.fit(x_train, y_train.values.ravel())
rf.score(x_train, y_train)
rf.score(x_test, y_test)
# Predicted Values and MSE

y_pred_train = rf.predict(x_train)

print(mean_squared_error(y_train, y_pred_train))
# Training on entire set and predict values.



rf.fit(x, y_case.values.ravel())
# Predicted ConfirmedCases

rf_pred_case = rf.predict(data_test_clean)
plt.figure(figsize=(15,8))

plt.plot(rf_pred_case)
rf.fit(x, y_fatal.values.ravel())
rf_pred_fatal = rf.predict(data_test_clean)
plt.figure(figsize=(20,8))

plt.plot(rf_pred_fatal)
# Saving to Submission.csv



#submission = pd.read_csv(path_sbumit)

#submission['ConfirmedCases'] = rf_pred_case

#submission['Fatalities'] = rf_pred_fatal



#submission.to_csv('submission.csv', index = False)
import xgboost as xgb

from sklearn.metrics import mean_squared_error
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
reg_y_pred = reg.predict(x_train)
mean_squared_error(y_train, reg_y_pred)
reg.score(x_test, y_test)
# Slightly Better than Random Forest 

reg_y_test_pred = reg.predict(x_test)

mean_squared_error(y_test, reg_y_test_pred)
reg.fit(x, y_case)
y_train_pred = reg.predict(x)
plt.plot(y_case)
plt.plot(y_train_pred)
mean_squared_error(y_case, y_train_pred)
xgb_pred_case = reg.predict(data_test_clean)
plt.plot(xgb_pred_case)
reg.fit(x, y_fatal)
# Checking MSE for Fatalities



print(mean_squared_error(y_fatal, reg.predict(x)))
plt.plot(reg.predict(x))
plt.plot(y_fatal)
xgb_pred_fatal = reg.predict(data_test_clean)
plt.plot(xgb_pred_fatal)
# Saving to Submission.csv



submission = pd.read_csv(submit_kaggle)

submission['ConfirmedCases'] = xgb_pred_case

submission['Fatalities'] = xgb_pred_fatal



submission.to_csv('submission.csv', index = False)