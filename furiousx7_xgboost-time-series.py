import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import imageio
import os
from statsmodels.graphics.tsaplots import plot_acf
#Data consists of two columns: date-time and consumption for one hour.

energy_hourly = pd.read_csv('../input/hourly-energy-consumption/PJME_hourly.csv', 
                            index_col=[0], parse_dates=[0])

#Indices are not sorted - order the readings
energy_hourly.sort_index(inplace=True)

#PJME_MW - MW per hour in PJM East Area
energy_hourly.head(3)
def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()
train, test = split_data(energy_hourly, '01-Jan-2015')

plt.figure(figsize=(15,5))
plt.xlabel('time')
plt.ylabel('energy consumed')
plt.plot(train.index,train)
plt.plot(test.index,test)
plt.show()
def create_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X
X_train, y_train = create_features(train), train['PJME_MW']
X_test, y_test   = create_features(test), test['PJME_MW']

X_train.shape, y_train.shape
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=False) # Change verbose to True if you want to see it train
xgb.plot_importance(reg, height=0.9)
def plot_performance(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(15,3))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('time')
    plt.ylabel('energy consumed')
    plt.plot(energy_hourly.index,energy_hourly, label='data')
    plt.plot(X_test.index,X_test_pred, label='prediction')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
X_test_pred = reg.predict(X_test)
    
plot_performance(energy_hourly, energy_hourly.index[0].date(), energy_hourly.index[-1].date(),
                 'Original and Predicted Data')

plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(),
                 'Test and Predicted Data')

plot_performance(y_test, '01-01-2015', '02-01-2015', 'January 2015 Snapshot')

plt.legend()

plt.show()
random_weeks = X_test[['year', 'weekofyear']].sample(10)
for week in random_weeks.iterrows():
    index = (X_test.year == week[1].year) & \
            (X_test.weekofyear == week[1].weekofyear)
    data = y_test[index]
    plot_performance(data, data.index[0].date(), data.index[-1].date())
mean_squared_error(y_true=y_test,
                   y_pred=X_test_pred)
mean_absolute_error(y_true=y_test,
                   y_pred=X_test_pred)
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_test,X_test_pred)
error_by_week = []
random_weeks = X_test[['year', 'weekofyear']].sample(10)
for week in random_weeks.iterrows():
    index = (X_test.year == week[1].year) & \
            (X_test.weekofyear == week[1].weekofyear)
    error_by_week.append(mean_absolute_percentage_error(y_test[index], X_test_pred[index]))
pd.Series(error_by_week, index=random_weeks.index)
X_test['PJME_MW'] = y_test
X_test['MW_Prediction'] = X_test_pred
X_test['error'] = y_test - X_test_pred
X_test['abs_error'] = X_test['error'].apply(np.abs)
error_by_day = X_test.groupby(['year','month','dayofmonth']) \
   .mean()[['PJME_MW','MW_Prediction','error','abs_error']]

error_by_day.sort_values('error', ascending=True).head(10)
# Best predicted days
error_by_day.sort_values('abs_error', ascending=True).head(10)
series = pd.Series.from_csv('../input/hourly-energy-consumption/PJME_hourly.csv', header=0)
plot_series = series[series.index<pd.Timestamp(series.index[1].date())]
plot_acf(series[0:24])
datetime.datetime
