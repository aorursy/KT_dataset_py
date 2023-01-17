import pandas as pd

import numpy as np

from pandas import datetime

import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

%matplotlib inline
consumption = pd.read_csv('/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt', sep = ';', parse_dates= ['Date'], infer_datetime_format=True, low_memory=False,  na_values=['nan','?'])
consumption.head()
consumption.describe()
consumption.info()
consumption.isna().sum()
consumption = consumption.dropna()

consumption.isna().sum()
mean_consumption_gby_date = consumption.groupby(['Date']).mean()
fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_gby_date.columns

axs[0, 0].plot(mean_consumption_gby_date[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_gby_date[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_gby_date[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_gby_date[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)



axs[2, 0].plot(mean_consumption_gby_date[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_gby_date[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_gby_date[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)
mean_consumption_gby_month = consumption.groupby(consumption['Date'].dt.strftime('%B')).mean()

reorderlist = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December' ]

mean_consumption_gby_month = mean_consumption_gby_month.reindex(reorderlist)
fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_gby_month.columns



axs[0, 0].plot(mean_consumption_gby_month[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_gby_month[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_gby_month[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_gby_month[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)





axs[2, 0].plot(mean_consumption_gby_month[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_gby_month[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_gby_month[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)
import pandas as pd

import numpy as np

from pandas import datetime as dt

import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()
consumption_2 = pd.read_csv('/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt', sep=';', 

                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 

                 low_memory=False, na_values=['nan','?'], index_col='dt')
mean_consumption_gby_day_month = consumption_2.groupby(consumption_2.index.day).mean()
fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_gby_day_month.columns



axs[0, 0].plot(mean_consumption_gby_day_month[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_gby_day_month[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_gby_day_month[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_gby_day_month[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)



axs[2, 0].plot(mean_consumption_gby_day_month[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_gby_day_month[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_gby_day_month[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

mean_consumption_gby_day_week = consumption_2.groupby(consumption_2.index.day_name()).mean().reindex(days)
fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_gby_day_week.columns



axs[0, 0].plot(mean_consumption_gby_day_week[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_gby_day_week[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_gby_day_week[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_gby_day_week[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)



axs[2, 0].plot(mean_consumption_gby_day_week[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_gby_day_week[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_gby_day_week[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)
consumption_resampled_in_a_day = consumption_2.resample('H').sum()

consumption_resampled_in_a_day.index = consumption_resampled_in_a_day.index.time

mean_consumption_gby_time = consumption_resampled_in_a_day.groupby(consumption_resampled_in_a_day.index).mean()

fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_gby_time.columns



axs[0, 0].plot(mean_consumption_gby_time[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_gby_time[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_gby_time[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_gby_time[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)



axs[2, 0].plot(mean_consumption_gby_time[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_gby_time[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_gby_time[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)

mean_consumption_resampled_mnthly = consumption_2.resample('M').mean()
fig, axs = plt.subplots(3, 2, figsize = (30, 25))

columns = mean_consumption_resampled_mnthly.columns



axs[0, 0].plot(mean_consumption_resampled_mnthly[columns[0]])

axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)



axs[0, 1].plot(mean_consumption_resampled_mnthly[columns[1]])

axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)



axs[1, 0].plot(mean_consumption_resampled_mnthly[columns[2]])

axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)



axs[1, 1].plot(mean_consumption_resampled_mnthly[columns[3]])

axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)



axs[2, 0].plot(mean_consumption_resampled_mnthly[columns[4]])

axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)



axs[2, 1].plot(mean_consumption_resampled_mnthly[columns[5]])

axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)



fig, axs = plt.subplots( figsize = (20, 4))

axs.plot(mean_consumption_resampled_mnthly[columns[6]])

axs.set_title(columns[6], fontweight = 'bold', size = 15)



from statsmodels.tsa.stattools import adfuller

def adf_test(ts, signif=0.05):

    dftest = adfuller(ts, autolag='AIC')

    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])

    for key,value in dftest[4].items():

       adf['Critical Value (%s)'%key] = value

    print (adf)

    

    p = adf['p-value']

    if p <= signif:

        print(f" Series is Stationary")

    else:

        print(f" Series is Non-Stationary")
adf_test(mean_consumption_resampled_mnthly["Global_active_power"])
adf_test(mean_consumption_resampled_mnthly['Global_reactive_power'])
adf_test(mean_consumption_resampled_mnthly['Voltage'])
adf_test(mean_consumption_resampled_mnthly['Global_intensity'])
adf_test(mean_consumption_resampled_mnthly['Sub_metering_1'])
adf_test(mean_consumption_resampled_mnthly['Sub_metering_2'])
adf_test(mean_consumption_resampled_mnthly['Sub_metering_3'])
def difference(dataset, interval=1):

    diff = list()

    diff.append(0)

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return diff



mean_consumption_resampled_mnthly['Voltage'] = difference(mean_consumption_resampled_mnthly['Voltage'])

adf_test(mean_consumption_resampled_mnthly['Voltage'])
model = VAR(mean_consumption_resampled_mnthly)

model_fit = model.fit()

pred = model_fit.forecast(model_fit.y, steps=4)