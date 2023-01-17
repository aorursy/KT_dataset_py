%matplotlib  inline



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import glob

import missingno as msno

from fbprophet import Prophet



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

print(os.listdir("../input"))
colors = ["windows blue", "amber", "faded green", "dusty purple"]

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 

            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })
path =r'../input/csvs_per_year/csvs_per_year/' # use your path

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
msno.matrix(frame);
msno.bar(frame);
stations = pd.read_csv('../input/stations.csv')

stations.head(1)
locations  = stations[['lat', 'lon']]

locationlist = locations.values.tolist()



popup = stations[['name']]



import folium

map_osm = folium.Map(location=[40.44, -3.69],

                    # tiles='Stamen Toner',

                     zoom_start=11) 



for point in range(0, len(locationlist)):

    folium.Marker(locationlist[point], popup=popup.iloc[point,0]).add_to(map_osm)

    

map_osm
cols = ['date', 'station', 'O_3']

o3 = frame[cols]



o3['date'] = pd.to_datetime(o3['date'])

o3['ppb'] = 24.45*o3['O_3'] /48



o3.head()
# active stations with time

plt.plot(o3.groupby(['date']).station.nunique());
# non-nulls per station

count_rows = pd.DataFrame(o3.groupby(['station']).O_3.count())

top3 = count_rows.sort_values('O_3', ascending=False).head(3)

top3
stations[stations.id == top3.index[0]]
# select station with most data

o3_station = o3[o3.station == top3.index[0]]



# Calculate Eight-Hour Average Ozone Concentrations

o3_station['ppb_rolling'] = o3_station['ppb'].rolling(8).mean()



del o3_station['station']

del o3_station['O_3']

del o3_station['ppb']



o3_station = o3_station.sort_values("date")



o3_station.columns = ['ds', 'y']

o3_station.set_index('ds', inplace=True)



# Resample to daily max

o3_station = o3_station.resample('D', how='max')



# Any missing dates?

d = pd.DataFrame(pd.date_range(start= o3_station.index.min(), end= o3_station.index.max(), freq='D'))   



o3_station.reset_index(level=0, inplace=True)

o3_station = d.join(o3_station)

del o3_station['ds']
o3_station.columns = ['ds', 'y']



# fill na 

o3_station['y'].fillna(0, inplace=True)
# 3 years in hours

i = 3*365 



# Train test split 

train = o3_station[1:-i]

test = o3_station[-i:]



train.info()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



ax = sns.scatterplot(x=train.index, y=train.y)

ax = sns.scatterplot(x=test.index, y=test.y)



ax.axes.set_xlim(train.index.min(), test.index.max());
train.reset_index(level=0, inplace=True)

test.reset_index(level=0, inplace=True)



train['y'] = np.log1p(train.y)

test['y'] = np.log1p(test.y)



train.head()
%%time



m = Prophet(changepoint_prior_scale=0.01) 

m.fit(train)
%%time

future = m.make_future_dataframe(periods=i, freq='D')



forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from  fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig2 = m.plot_components(forecast);
test = pd.concat([test.set_index('ds'),forecast.set_index('ds')], axis=1, join='inner')



cols = ['y', 'yhat', 'yhat_lower', 'yhat_upper']

test = test[cols]

#test['y'] = np.expm1(test.y)

#test['yhat'] = np.expm1(test.yhat)

#test['yhat_lower'] = np.expm1(test.yhat_lower)

#test['yhat_upper'] = np.expm1(test.yhat_upper)



test.tail()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



plt.plot(test.y)

plt.plot(test.yhat)

plt.legend();
test['e'] = test.y - test.yhat



rmse = np.round(np.sqrt(np.mean(test.e**2)),2)

mape = np.round(np.mean(np.abs(100*test.e/test.y)), 2)

print('RMSE =', rmse)

print('MAPE =', mape, '%')
cols = ['ds', 'y']

train = train[cols]



train.set_index('ds', inplace=True)

train.head()
test.head()
colors = ["windows blue", "amber", "faded green", "dusty purple"]

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 

            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })
seasonal_decompose(train.y, model='additive').plot()

print("Dickey–Fuller test: p=%f" % adfuller(train.y)[1])
# Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots

ax = plt.subplot(211)



# Plot the autocorrelation function

plot_acf(train.y[0:].values.squeeze(), lags=750, ax=ax)

ax = plt.subplot(212)

plot_pacf(train.y[0:].values.squeeze(), lags=750, ax=ax)

plt.tight_layout()
# Initial approximation of parameters

ps = range(0, 2)

d = 1

qs = range(0, 2)



parameters = product(ps, qs)

parameters_list = list(parameters)

len(parameters_list)
%%time 



# Model Selection

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
best_model.plot_diagnostics(figsize=(15, 12))

plt.show()
test['yhat_ARIMA'] = best_model.forecast(test.shape[0])

test.tail()
np.expm1(test.y).plot(linewidth=3)

np.expm1(test.yhat_ARIMA).plot(color='r', ls='--', label='Predicted Units', linewidth=3)



plt.legend()

plt.grid()

plt.title('Max 8-hour average ozone concentration - daily forecast')

plt.ylabel('parts per billion');
test['e'] = test.y - test.yhat_ARIMA



rmse = np.round(np.sqrt(np.mean(test.e**2)),2)

mape = np.round(np.mean(np.abs(100*test.e/test.y)), 2)

print('RMSE =', rmse)

print('MAPE =', mape, '%')
test.tail()
np.expm1(test.y).plot(linewidth=3)



np.expm1(test.yhat_ARIMA).plot(color='grey', ls='--', label='ARIMA forecast', linewidth=3)

np.expm1(test.yhat).plot(color='r', ls='--', label='Prophet forecast', linewidth=3)



plt.legend()

plt.grid()

plt.title('Max 8-hour average ozone concentration - daily forecast')

plt.ylabel('parts per billion');