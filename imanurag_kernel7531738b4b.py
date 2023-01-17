# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas

import matplotlib.pyplot as plt

from matplotlib import pyplot

import plotly.express as px

import plotly.graph_objects as go

from urllib.request import urlopen

import json

from dateutil.parser import parse

import math

import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter

from datetime import datetime

from statsmodels.tsa.arima_model import ARIMA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates = ['Date'])

data = data[data['ConfirmedCases']!=0]

data.loc[data['Country/Region'] == 'US', ['Country/Region']] = 'United States of America'
gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.Long, data.Lat))
gdf_CC = data.groupby('Country/Region')['ConfirmedCases'].max()

data = data[data['Fatalities']!=0]

gdf_fatalities = data.groupby('Country/Region')['Fatalities'].max()

gdf_fatalities, gdf_CC
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

ax = world.plot(figsize = (20,30))

gdf.plot(ax = ax, column = 'Fatalities', marker = "o", markersize = 3)

plt.show()
with open('/kaggle/input/world-countries/world-countries.json') as f:

    countries = json.load(f)
gdf_CC = pd.DataFrame(gdf_CC)

gdf_CC['Confirmed Cases LogScale'] = np.log(gdf_CC['ConfirmedCases'])

gdf_CC.reset_index(inplace = True)

gdf_CC['ConfirmedCasesStr'] = gdf_CC['ConfirmedCases'].astype(str)

gdf_CC['CC'] = "\n" + gdf_CC['ConfirmedCasesStr']
fig = go.Figure(go.Choropleth(

                geojson = countries,

                z = gdf_CC['Confirmed Cases LogScale'],

                text = gdf_CC['CC'], 

                locations = gdf_CC['Country/Region'], 

                featureidkey = "properties.name", 

                autocolorscale=False,

                colorbar_title="Confirmed Cases LogScale"

))



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
gdf_fatalities = pd.DataFrame(gdf_fatalities)

gdf_fatalities['Fatalities LogScale'] = np.log(gdf_fatalities['Fatalities'])

gdf_fatalities.reset_index(inplace = True)

gdf_fatalities['FatalitiesStr'] = gdf_fatalities['Fatalities'].astype(str)

gdf_fatalities['FF'] = "\n" + gdf_fatalities['FatalitiesStr']
fig = go.Figure(go.Choropleth(

                geojson = countries,

                z = gdf_fatalities['Fatalities LogScale'],

                locations = gdf_fatalities['Country/Region'], 

                featureidkey = "properties.name", 

                autocolorscale=False,

                text = gdf_fatalities['FF'],

                colorbar_title="Fatalities LogScale"

))



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
data_ts = data[['Country/Region','Date','ConfirmedCases', 'Fatalities']]

data_ts_world_CC = data_ts.groupby('Date')['ConfirmedCases'].sum()

data_ts_world_FF = data_ts.groupby('Date')['Fatalities'].sum()

data_ts_world_CC = pd.DataFrame(data_ts_world_CC)

data_ts_world_FF = pd.DataFrame(data_ts_world_FF)

data_ts_world_CC.reset_index(inplace = True)

data_ts_world_FF.reset_index(inplace = True)

data_ts_world_CC.loc[0, 'NewCases'] = 0

for i in range(1,len(data_ts_world_CC)):

    data_ts_world_CC.loc[i,'NewCases'] = data_ts_world_CC.loc[i, 'ConfirmedCases'] - data_ts_world_CC.loc[i-1, 'ConfirmedCases']

data_ts_world_FF.loc[0, 'NewFatalities'] = 0

for i in range(1,len(data_ts_world_FF)):

    data_ts_world_FF.loc[i,'NewFatalities'] = data_ts_world_FF.loc[i, 'Fatalities'] - data_ts_world_FF.loc[i-1, 'Fatalities']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=100)

ax1.set(title='ConfirmedCases everyday', xlabel='Date', ylabel='ConfirmedCases')

date_form = DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(date_form)

ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))

ax1.xaxis.set_tick_params(rotation=45)



ax2.set(title='NewCases everyday', xlabel='Date', ylabel='NewCases')

date_form = DateFormatter("%Y-%m-%d")

ax2.xaxis.set_major_formatter(date_form)

ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))

ax2.xaxis.set_tick_params(rotation=45)



ax1.plot(data_ts_world_CC.Date, data_ts_world_CC.ConfirmedCases, color='tab:red')

ax2.plot(data_ts_world_CC.Date, data_ts_world_CC.NewCases, color='tab:red')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=100)

ax1.set(title='Fatalities everyday', xlabel='Date', ylabel='Fatalities')

date_form = DateFormatter("%Y-%m-%d")

ax1.xaxis.set_major_formatter(date_form)

ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))

ax1.xaxis.set_tick_params(rotation=45)



ax2.set(title='NewFatalities everyday', xlabel='Date', ylabel='NewFatalities')

date_form = DateFormatter("%Y-%m-%d")

ax2.xaxis.set_major_formatter(date_form)

ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))

ax2.xaxis.set_tick_params(rotation=45)



ax1.plot(data_ts_world_FF.Date, data_ts_world_FF.Fatalities, color='tab:red')

ax2.plot(data_ts_world_FF.Date, data_ts_world_FF.NewFatalities, color='tab:red')
#Top 5 countries with confirmed cases

gdf_CC.nlargest(10, 'ConfirmedCases')[['Country/Region', 'ConfirmedCases']]
gdf_fatalities.nlargest(10, 'Fatalities')[['Country/Region', 'Fatalities']]
data_ts_arima = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', 

                            usecols = ['Date', 'ConfirmedCases'], header=0, parse_dates=[0], squeeze=True, date_parser=parse)

data_ts_arima_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', 

                            usecols = ['Date'], header=0, parse_dates=[0], squeeze=True, date_parser=parse)
data_ts_arima = data_ts_arima[data_ts_arima['ConfirmedCases']!=0]

data_ts_world_arima = data_ts_arima.groupby('Date')['ConfirmedCases'].sum()

X = data_ts_world_arima.values

size = int(len(X) * 0.8)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = ARIMA(history, order=(5,1,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

# plot

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()