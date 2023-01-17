import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd
os.chdir(r'../input/')



#train_url = r'https://www.kaggle.com/c/bike-sharing-demand/download/train.csv'

#test_url = r'https://www.kaggle.com/c/bike-sharing-demand/download/train.csv'



df_train = pd.read_csv('train.csv')

df_train.head()

def null_percentage(column):

    df_name = column.name

    nans = np.count_nonzero(column.isnull().values)

    total = column.size

    frac = nans / total

    perc = int(frac * 100)

    print('%d%% of values or %d missing from %s column.' % (perc, nans, df_name))



def check_null(df, columns):

    for col in columns:

        null_percentage(df[col])

        

check_null(df_train, df_train.columns)
df_train['month'] = df_train.apply(lambda x: x['datetime'][5:7], axis=1).astype(int)

df_train['hour'] = df_train.apply(lambda x: x['datetime'][11:13], axis=1).astype(int)
median_temp = df_train.atemp.median()

df_train['temp_dev'] = df_train.apply(lambda x: x['atemp'] - median_temp, axis=1)
#Ridership by Month

    

plt.figure('Daily rides by month', figsize=(10, 15))

plt.subplot(211)

sns.boxplot(x='month', y='count', hue='workingday', data=df_train)



plt.subplot(212)

sns.boxplot(x='month', y='count', hue='holiday', data=df_train)

plt.show()



plt.figure('Casual', figsize=(10, 15))

plt.subplot(211)

sns.boxplot(x='month', y='casual', hue='workingday', data=df_train)



plt.subplot(212)

sns.boxplot(x='month', y='casual', hue='holiday', data=df_train)

plt.show()



plt.figure('Registered', figsize=(10, 15))

plt.subplot(211)

sns.boxplot(x='month', y='registered', hue='workingday', data=df_train)



plt.subplot(212)

sns.boxplot(x='month', y='registered', hue='holiday', data=df_train)

plt.show()
df_train.loc[(df_train.month == 4) & (df_train.holiday == 1), 'holiday_4'] = 1

df_train.loc[df_train.holiday_4.isnull(), 'holiday_4'] = 0



df_train.loc[(df_train.month == 7) & (df_train.holiday == 1), 'holiday_7'] = 1

df_train.loc[df_train.holiday_7.isnull(), 'holiday_7'] = 0



df_train.loc[(df_train.month == 9) & (df_train.holiday == 1), 'holiday_9'] = 1

df_train.loc[df_train.holiday_9.isnull(), 'holiday_9'] = 0



df_train.loc[(df_train.month == 10) & (df_train.holiday == 1), 'holiday_10'] = 1

df_train.loc[df_train.holiday_10.isnull(), 'holiday_10'] = 0



df_train.loc[(df_train.month == 11) & (df_train.holiday == 1), 'holiday_11'] = 1

df_train.loc[df_train.holiday_11.isnull(), 'holiday_11'] = 0
plt.figure('Total Rides by Season', figsize=(10, 15))

plt.subplot(311)

sns.boxplot(x='season', y='count', hue='workingday', data=df_train)



plt.subplot(312)

sns.boxplot(x='season', y='casual', hue='workingday', data=df_train)



plt.subplot(313)

sns.boxplot(x='season', y='registered', hue='workingday', data=df_train)



plt.show()
plt.figure('Ridership v Temp', figsize=(10, 15))

plt.subplot(311)

sns.regplot(x='temp_dev', y='count', data=df_train, x_bins=10, order=2)

plt.subplot(312)

sns.regplot(x='temp_dev', y='casual', data=df_train, x_bins=10, order=2)

plt.subplot(313)

sns.regplot(x='temp_dev', y='registered', data=df_train, x_bins=10, order=2)

plt.show()
plt.figure('Ridership v Actual Temp', figsize=(10, 15))

plt.subplot(311)

sns.regplot(x='temp', y='count', data=df_train, x_bins=10, order=2)

plt.subplot(312)

sns.regplot(x='temp', y='casual', data=df_train, x_bins=10, order=2)

plt.subplot(313)

sns.regplot(x='temp', y='registered', data=df_train, x_bins=10, order=2)

plt.show() 
plt.figure('Ridership v Humidity', figsize=(10, 15))

plt.subplot(311)

sns.regplot(x='humidity', y='count', data=df_train, x_bins=10, order=2)

plt.subplot(312)

sns.regplot(x='humidity', y='casual', data=df_train, x_bins=10, order=2)

plt.subplot(313)

sns.regplot(x='humidity', y='registered', data=df_train, x_bins=10, order=2)

plt.show() 
plt.figure('Ridership v Wind', figsize=(10, 15))

plt.subplot(311)

sns.regplot(x='windspeed', y='count', data=df_train, x_bins=15, order=3)

plt.subplot(312)

sns.regplot(x='windspeed', y='casual', data=df_train, x_bins=15, order=3)

plt.subplot(313)

sns.regplot(x='windspeed', y='registered', data=df_train, x_bins=15, order=3)

plt.show() 
plt.figure('Wind by month')

sns.boxplot(x='month', y='windspeed', data=df_train)

plt.show()
df_train.weather.value_counts()
plt.figure('Weather and Ridership', figsize=(10, 15))

plt.subplot(311)

sns.boxplot(x='weather', y='count', data=df_train)

plt.subplot(312)

sns.boxplot(x='weather', y='casual', data=df_train)

plt.subplot(313)

sns.boxplot(x='weather', y='registered', data=df_train)

plt.show()
df_train.loc[df_train['weather'] == 4, 'weather'] = 3

df_train.weather.value_counts()
def heatmap(df):

    plt.figure('heatmap')

    df_corr = df.corr()

    sns.heatmap(df_corr, vmax=0.6, square=True, annot=False)

    plt.yticks(rotation = 0)

    plt.xticks(rotation = 90)

    plt.show()

    

heatmap(df_train)
import pandas as pd

df_train = pd.DataFrame()

df_train = pd.read_csv('train.csv', header=0)



df = df_train.copy()

df_test = pd.read_csv('test.csv', header=0)



df_train['train'] = 1

df_test['train'] = 0

df = pd.concat([df_train, df_test], ignore_index=False, axis=0) 

del(df_test, df_train)



#check_null(df, df.columns)
df['month'] = df.apply(lambda x: x['datetime'][5:7], axis=1).astype(int)

df['hour'] = df.apply(lambda x: x['datetime'][11:13], axis=1).astype(int)



median_temp = df.atemp.median()

df['temp_dev'] = df.apply(lambda x: x['atemp'] - median_temp, axis=1)



df.loc[(df.month == 4) & (df.holiday == 1), 'holiday_4'] = 1

df.loc[df.holiday_4.isnull(), 'holiday_4'] = 0

df.loc[(df.month == 7) & (df.holiday == 1), 'holiday_7'] = 1

df.loc[df.holiday_7.isnull(), 'holiday_7'] = 0

df.loc[(df.month == 9) & (df.holiday == 1), 'holiday_9'] = 1

df.loc[df.holiday_9.isnull(), 'holiday_9'] = 0

df.loc[(df.month == 10) & (df.holiday == 1), 'holiday_10'] = 1

df.loc[df.holiday_10.isnull(), 'holiday_10'] = 0

df.loc[(df.month == 11) & (df.holiday == 1), 'holiday_11'] = 1

df.loc[df.holiday_11.isnull(), 'holiday_11'] = 0
columns_used = ['weather'

                #'datetime'

                , 'season'

                #, 'holiday'

                , 'workingday'

                #, 'month'

                #, 'temp'

                #, 'atemp'

                #, 'humidity'

                #, 'windspeed'

                , 'casual'

                , 'registered'

                #, 'count'

                , 'hour'

                , 'temp_dev'

                #, 'holiday_4'

                #, 'holiday_7'

                #, 'holiday_9'

                #, 'holiday_11'

                #, 'holiday_10'

                , 'training']
df_submission = df.loc[df['train'] == 0]

df_submission = df_submission.drop(['train', 'casual', 'registered', 'count'], axis=1)

X_submission = df_submission.drop(['datetime'], axis=1)



df = df.loc[df['train'] == 1]

y_casual = df['casual']

y_count = df['count']

y_reg = df['registered']



X = df.drop(['datetime', 'train', 'casual', 'registered', 'count'], axis=1)
from sklearn.model_selection import train_test_split



Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_casual, test_size=0.2)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2)
''' Feature Scaling ''' 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

Xc_train = sc_X.fit_transform(Xc_train)

Xr_train = sc_X.fit_transform(Xr_train)

Xc_test = sc_X.fit_transform(Xc_test)

Xr_test = sc_X.fit_transform(Xr_test)

X_submission = sc_X.fit_transform(X_submission)
# Random Forest





from sklearn.ensemble import RandomForestRegressor

cRegRF = RandomForestRegressor(n_estimators=50, bootstrap=True)

rRegRF = RandomForestRegressor(n_estimators=50, bootstrap=True)

cRegRF.fit(Xc_train, yc_train)

rRegRF.fit(Xr_train, yr_train)

yc_pred = cRegRF.predict(Xc_test)

yr_pred = rRegRF.predict(Xr_test)

def RMSLE(y_true, y_pred):

    sum_val = 0

    for true, pred in zip(y_true, y_pred):

        sum_val += (np.log(pred + 1) - np.log(true + 1)) ** 2

    sum_val = sum_val / len(y_true)

    return np.sqrt(sum_val)



y_pred_tot = np.array(yc_pred) + np.array(yr_pred)

y_true_tot = np.array(yc_test) + np.array(yr_test)



print('casual')

print(RMSLE(np.array(yc_test), np.array(yc_pred)))

print('registered')

print(RMSLE(np.array(yr_test), np.array(yr_pred)))

print('total')

print(RMSLE(y_true_tot, y_pred_tot))
