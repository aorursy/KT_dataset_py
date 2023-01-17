import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



import seaborn as sns # for plot visualization

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller, acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# import os

# print(os.listdir("../input"))
weather_df = pd.read_csv('../input/testset.csv', parse_dates=['datetime_utc'], index_col='datetime_utc')

weather_df.head()
weather_df = weather_df.loc[:,[' _conds', ' _hum', ' _tempm']]

weather_df = weather_df.rename(index=str, columns={' _conds': 'condition', ' _hum': 'humidity', ' _pressurem': 'pressure', ' _tempm': 'temprature'})

print(f'dataset shape (rows, columns) - {weather_df.shape}')

weather_df.head()
# lets check dtype of all columns, 

weather_df.dtypes, weather_df.index.dtype
weather_df.index = pd.to_datetime(weather_df.index)

weather_df.index
def list_and_visualize_missing_data(dataset):

    # Listing total null items and its percent with respect to all nulls

    total = dataset.isnull().sum().sort_values(ascending=False)

    percent = ((dataset.isnull().sum())/(dataset.isnull().count())).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    missing_data = missing_data[missing_data.Total > 0]

    

    missing_data.plot.bar(subplots=True, figsize=(16,9))



list_and_visualize_missing_data(weather_df)
# will fill with previous valid value

weather_df.ffill(inplace=True)

weather_df[weather_df.isnull()].count()
weather_df.describe()
weather_df = weather_df[weather_df.temprature < 50]

weather_df = weather_df[weather_df.humidity <= 100]
weather_condition = (weather_df.condition.value_counts()/(weather_df.condition.value_counts().sum()))*100

weather_condition.plot.bar(figsize=(16,9))

plt.xlabel('Weather Conditions')

plt.ylabel('Percent')
weather_df.plot(subplots=True, figsize=(20,12))
weather_df['2015':'2016'].resample('D').fillna(method='pad').plot(subplots=True, figsize=(20,12))
train_df = weather_df['2000':'2015'].resample('M').mean().fillna(method='pad')

train_df.drop(columns='humidity', axis=1, inplace=True)

test_df = weather_df['2016':'2017'].resample('M').mean().fillna(method='pad')

test_df.drop(columns='humidity', axis=1, inplace=True)
# check rolling mean and rolling standard deviation

def plot_rolling_mean_std(ts):

    rolling_mean = ts.rolling(12).mean()

    rolling_std = ts.rolling(12).std()

    plt.figure(figsize=(22,10))



    plt.plot(ts, label='Actual Mean')

    plt.plot(rolling_mean, label='Rolling Mean')

    plt.plot(rolling_std, label = 'Rolling Std')

    plt.xlabel("Date")

    plt.ylabel("Mean Temperature")

    plt.title('Rolling Mean & Rolling Standard Deviation')

    plt.legend()

    plt.show()
# Augmented Dickey–Fuller test

def perform_dickey_fuller_test(ts):

    result = adfuller(ts, autolag='AIC')

    print('Test statistic: ' , result[0])

    print('Critical Values:' ,result[4])
# check stationary: mean, variance(std)and adfuller test

plot_rolling_mean_std(train_df.temprature)

perform_dickey_fuller_test(train_df.temprature)
# Original Series

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})



fig, axes = plt.subplots(3, 2, sharex=True)

axes[0, 0].plot(train_df.values); 

axes[0, 0].set_title('Original Series')

plot_acf(train_df.values, ax=axes[0, 1])



# 1st Differencing

axes[1, 0].plot(train_df.temprature.diff().values); 

axes[1, 0].set_title('1st Order Differencing')

plot_acf(train_df.diff().dropna().values,ax=axes[1, 1])



# 2nd Differencing

axes[2, 0].plot(train_df.temprature.diff().diff().values); 

axes[2, 0].set_title('2nd Order Differencing')

plot_acf(train_df.diff().diff().dropna().values,ax=axes[2, 1])



plt.xticks(rotation='vertical')

plt.show()
# PACF plot of 1st differenced series

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})



fig, axes = plt.subplots(1, 2, sharex=True)

axes[0].plot(train_df.diff().values); axes[0].set_title('1st Differencing')

axes[1].set(ylim=(0,5))

plot_pacf(train_df.diff().dropna().values, ax=axes[1])



plt.show()
fig, axes = plt.subplots(1, 2, sharex=True)

axes[0].plot(train_df.diff().values); axes[0].set_title('1st Differencing')

axes[1].set(ylim=(0,1.2))

plot_acf(train_df.diff().dropna().values, ax=axes[1])



plt.show()
acf_lag = acf(train_df.diff().dropna().values, nlags=20)

pacf_lag = pacf(train_df.diff().dropna().values, nlags=20, method='ols')



plt.figure(figsize=(22,10))



plt.subplot(121)

plt.plot(acf_lag)

plt.axhline(y=0,linestyle='--',color='silver')

plt.axhline(y=-1.96/np.sqrt(len(train_df.diff().values)),linestyle='--',color='silver')

plt.axhline(y=1.96/np.sqrt(len(train_df.diff().values)),linestyle='--',color='silver')

plt.title("Autocorrelation Function")



plt.subplot(122)

plt.plot(pacf_lag)

plt.axhline(y=0,linestyle='--',color='silver')

plt.axhline(y=-1.96/np.sqrt(len(train_df.diff().values)),linestyle='--',color='silver')

plt.axhline(y=1.96/np.sqrt(len(train_df.diff().values)),linestyle='--',color='silver')

plt.title("Partial Autocorrelation Function")

plt.tight_layout()
model = ARIMA(train_df.values, order=(2,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
# Actual vs Fitted

model_fit.plot_predict(dynamic=False)

plt.show()
# # Forecast

fc, se, conf = model_fit.forecast(16, alpha=0.05)  # 95% conf



# print(fc)

# Make as pandas series

fc_series = pd.Series(fc, index=test_df.index)

lower_series = pd.Series(conf[:, 0], index=test_df.index)

upper_series = pd.Series(conf[:, 1], index=test_df.index)



# # Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train_df, label='training')

plt.plot(test_df, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()

# test_df.index