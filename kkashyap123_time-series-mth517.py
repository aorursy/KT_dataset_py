%matplotlib  inline



import numpy as np 

import pandas as pd 

import seaborn as sns

sns.set()

import glob 

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import stats

import statsmodels.api as sm

from itertools import product

from math import sqrt

from sklearn.metrics import mean_squared_error 



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input/air-quality-madrid"))
path =r'../input/air-quality-madrid/csvs_per_year/csvs_per_year/' # use your path

allFiles = glob.glob(path + "/*.csv")

frame = pd.DataFrame()

list_ = []

for file_ in allFiles:

    df = pd.read_csv(file_,index_col=None, header=0)

    list_.append(df)

frame = pd.concat(list_)



cols = ['date', 'station', 'BEN', 'CH4', 'CO', 'EBE', 'MXY', 'NMHC', 'NO', 'NO_2', 'NOx', 'OXY',

       'O_3', 'PM10', 'PM25', 'PXY', 'SO_2', 'TCH', 'TOL']

frame = frame[cols]

frame = frame.sort_values(['station', 'date'])



frame.tail(3)
cols = ['date', 'station', 'O_3']

o3 = frame[cols]



o3['date'] = pd.to_datetime(o3['date'])

o3['ppb'] = 24.45*o3['O_3'] /48



o3.head()
count_rows = pd.DataFrame(o3.groupby(['station']).O_3.count())

top3 = count_rows.sort_values('O_3', ascending=False).head(3)

top3
stations = pd.read_csv('../input/air-quality-madrid/stations.csv')

stations[stations.id == top3.index[0]]

o3_station = o3[o3.station == top3.index[0]]

o3_station['ppb_rolling'] = o3_station['ppb'].rolling(8).mean()

o3_station.head()
del o3_station['station']

del o3_station['O_3']

del o3_station['ppb']

o3_station = o3_station.sort_values("date")



o3_station.columns = ['ds', 'y']

o3_station.set_index('ds', inplace=True)



o3_station = o3_station.resample('D', how='max')

o3_station.head()


d = pd.DataFrame(pd.date_range(start= o3_station.index.min(), end= o3_station.index.max(), freq='D'))   



o3_station.reset_index(level=0, inplace=True)

o3_station = d.join(o3_station)

del o3_station['ds']
o3_station.head()
o3_station.columns = ['ds', 'y']



o3_station['y'].fillna(0, inplace=True)

o3_station.head()
i = 3*365 



# Train test split 

train = o3_station[1:-i]

test = o3_station[-i:]

train['y'] = np.log1p(train.y)

test['y'] = np.log1p(test.y)

train.head()

cols = ['ds', 'y']

train = train[cols]

train.set_index('ds', inplace=True)



test=test[cols]

test.set_index('ds', inplace=True)

train.head()
#colors = ["windows blue", "amber", "faded green", "dusty purple"]

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 

            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })





seasonal_decompose(train.y, model='additive').plot()

print("Dickey–Fuller test: p=%f" % adfuller(train.y)[1])

ax = plt.subplot(211)



# Plot the autocorrelation function

plot_acf(train.y[0:].values.squeeze(), lags=750, ax=ax)

ax = plt.subplot(212)

plot_pacf(train.y[0:].values.squeeze(), lags=750, ax=ax)

plt.tight_layout()





ps = range(0, 2)

d = 1

qs = range(0, 2)

parameters = product(ps, qs)

parameters_list = list(parameters)



train.head()
results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')

for param in parameters_list:

    try:

        model = SARIMAX(train.y, order=(param[0], d, param[1])).fit(disp=-1)

    except ValueError:

        print('bad parameter combination:', param)

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])

# Best Models

result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']

print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())
print("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])
test.head()

best_model.forecast(test.shape[0])

test['yhat_ARIMA'] = best_model.forecast(test.shape[0])

test.head()
test['e'] = test.y - test.yhat_ARIMA

test.tail()
rmse = np.round(np.sqrt(np.mean(test.e**2)),2)

mape = np.round(np.mean(np.abs(100*test.e/test.y)), 2)

print('RMSE =', rmse)

print('MAPE =', mape, '%')

np.expm1(test.y).plot(linewidth=3)

np.expm1(test.yhat_ARIMA).plot(color='r', ls='--', label='Predicted Units', linewidth=3)



plt.legend()

plt.grid()

plt.title('Max 8-hour average ozone concentration - daily forecast')

plt.ylabel('parts per billion');