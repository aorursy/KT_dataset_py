import pandas as pd

import matplotlib

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
data_set = pd.read_excel('../input/MinDailyTemps.xlsx')
data_set.head(10)
data_set.isnull().sum()
data_set.dtypes
data_set = data_set.set_index('Date')
data_set.head().append(data_set.tail())
data_set.plot(grid=True)
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]

from datetime import datetime

start_date = datetime(1981, 1, 1)

end_date = datetime(1982, 12, 31)

data_set[(start_date <= data_set.index) & (data_set.index <= end_date)].plot(grid=True)
data_set.head()
decompfreq = 365 # for yearly seasonality
import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(data_set, 

                                          freq=decompfreq, 

                                          model = 'additive')

fig = decomposition.plot()

matplotlib.rcParams['figure.figsize'] = [9.0, 5.0]
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

fig, ax = plt.subplots()

ax.grid(True)

year = mdates.YearLocator(month=1)

month = mdates.MonthLocator(interval=3)

year_format = mdates.DateFormatter('%Y')

month_format = mdates.DateFormatter('%m')

ax.xaxis.set_minor_locator(month)

ax.xaxis.grid(True, which = 'minor')

ax.xaxis.set_major_locator(year)

ax.xaxis.set_major_formatter(year_format)

plt.plot(data_set.index, data_set['Temp'], c='blue')

plt.plot(decomposition.trend.index, decomposition.trend, c='red')