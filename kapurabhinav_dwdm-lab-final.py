import csv
import math
import random
import pandas as pd
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (10, 6)
day_ahead_market = pd.read_csv("../input/day_ahead_market_lbmp.csv")
real_time_market = pd.read_csv("../input/real_time_market_lbmp.csv")
print (day_ahead_market)
print(real_time_market)
day_ahead_market['Time Stamp'] = pd.to_datetime(day_ahead_market['Time Stamp'], format='%m/%d/%Y %H:%M')
real_time_market['Time Stamp'] = pd.to_datetime(real_time_market['Time Stamp'], format='%m/%d/%Y %H:%M:%S')

dam_time_name = day_ahead_market.set_index(['Name', 'Time Stamp'])
rtm_time_name = real_time_market.set_index(['Name', 'Time Stamp'])

dam_cap_lbmp = dam_time_name['LBMP ($/MWHr)']['CAPITL']
rtm_cap_lbmp = rtm_time_name['LBMP ($/MWHr)']['CAPITL']
print (dam_cap_lbmp)
print (rtm_cap_lbmp)
plt.figure(figsize=(10,8))
dam_cap_lbmp.plot(title='CAPITL Day Ahead LBMP')
plt.ylabel('Price ($)')
plt.show()
weekly = dam_cap_lbmp.resample('W').mean()
weekly.plot()
plt.ylabel('Weekly Day Ahead Price')
by_weekday = dam_cap_lbmp.groupby(dam_cap_lbmp.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
plt.ylabel('Daily Day Ahead Price')
by_weekday.plot()
weekend = np.where(dam_cap_lbmp.index.weekday < 5, 'Weekday', 'Weekend')
by_time = dam_cap_lbmp.groupby([weekend, dam_cap_lbmp.index]).mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays')
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends')
plt.figure(figsize=(10,8))
rtm_cap_lbmp.plot(title='CAPITL Realtime LBMP')
plt.ylabel('Price($)')
plt.show()
weekly = rtm_cap_lbmp.resample('W').mean()
weekly.plot()
plt.ylabel('Weekly Real Time Price')
by_weekday = rtm_cap_lbmp.groupby(rtm_cap_lbmp.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
plt.ylabel('Daily Real Time Price')
by_weekday.plot()
weekend = np.where(rtm_cap_lbmp.index.weekday < 5, 'Weekday', 'Weekend')
by_time = rtm_cap_lbmp.groupby([weekend, rtm_cap_lbmp.index]).mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays')
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends')
aligned_dam, aligned_rtm = rtm_cap_lbmp.align(dam_cap_lbmp, join='inner')
no_dup_al_dam = aligned_dam[~aligned_dam.index.duplicated(keep='first')]
no_dup_al_rtm = aligned_rtm[~aligned_dam.index.duplicated(keep='first')]

no_dup_al_dam.name = 'dam_lbmp'
no_dup_al_rtm.name = 'rtm_lbmp'

dam_rtm_df = pd.DataFrame([no_dup_al_dam, no_dup_al_rtm]).transpose()
forecast = pd.read_csv('../input/lbmp_w.csv')
print(forecast)
forecast['Forecast Date'] = pd.to_datetime(forecast['Forecast Date'], format='%m/%d/%Y')
forecast['Vintage Date'] = pd.to_datetime(forecast['Vintage Date'], format='%m/%d/%Y')
forecast['Vintage'] = forecast['Vintage'].astype('category')
lga_indexed = forecast[(forecast['Station ID'] == 'LGA')].set_index(['Forecast Date', 'Vintage Date','Vintage','Station ID'])
mean_cap_indexed = lga_indexed.mean(level=[0,1,2])
mean_cap = mean_cap_indexed.reset_index()
actual_temp_df = mean_cap[mean_cap['Vintage'] == 'Actual'].groupby(['Vintage Date']).first().rename(columns=lambda x: 'Actual ' + x)
dam_rtm_act_df = dam_rtm_df.join(actual_temp_df, how='left').fillna(method='ffill').dropna()
daily_df = dam_rtm_act_df.resample('D', how='mean')
plt.figure(figsize=(14,10))
plt.plot_date(daily_df.index, daily_df['rtm_lbmp'], '-', label='RTM LBMP')
plt.plot_date(daily_df.index, daily_df['dam_lbmp'], '-', label='DAM LBMP')
plt.plot_date(daily_df.index, daily_df['Actual Min Temp'], '-', label='Min Temp')
plt.plot_date(daily_df.index, daily_df['Actual Max Temp'], '-', label='Max Temp')
plt.legend()
plt.show()
exog_data = np.array([daily_df['Actual Max Temp'].values, daily_df['dam_lbmp'].values])
k = 250
m = ARIMA(daily_df['rtm_lbmp'].values[0:k], [0,0,1])
#m = ARIMA(daily_df['rtm_lbmp'].values[0:k], [0,0,1], exog=np.transpose(exog_data[:,0:k]), dates=daily_df.index.values[0:k])
results = m.fit(trend='nc', disp=True)
predicted_prices = results.predict(10, 364, exog=np.transpose(exog_data), dynamic=True)
plt.figure(figsize=(14, 10))
plt.plot(predicted_prices, label='prediction')
plt.plot(daily_df['rtm_lbmp'].values, label='actual RTM')
plt.legend()
plt.show()
plt.figure(figsize=(14, 10))
plt.plot(predicted_prices, label='prediction')
plt.plot(daily_df['rtm_lbmp'].values, label='actual RTM')
plt.plot(daily_df['dam_lbmp'].values, label='actual DAM')
plt.legend()
plt.show()