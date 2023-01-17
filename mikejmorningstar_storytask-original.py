import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots

import seaborn as sns

from pprint import pprint

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import fbprophet

from fbprophet.plot import plot_plotly, plot_components_plotly

import datetime

import os



df_pg1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_pg2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_ws1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_ws2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
# Data Preprocessing

## Here we make the DATE_TIME column (which right now contains string values) datetimeobjects

df_pg1['DATE_TIME'] = pd.to_datetime(df_pg1['DATE_TIME'], format = '%d-%m-%Y %H:%M')

df_pg2['DATE_TIME'] = pd.to_datetime(df_pg2['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_ws1['DATE_TIME'] = pd.to_datetime(df_ws1['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_ws2['DATE_TIME'] = pd.to_datetime(df_ws2['DATE_TIME'], format = '%Y-%m-%d %H:%M')

## Data Cleaning: Here we remove those columns which do not interest us

df_pg1 = df_pg1.drop(columns = 'PLANT_ID')

df_pg2 = df_pg2.drop(columns = 'PLANT_ID')

df_ws1 = df_ws1.drop(columns = ['PLANT_ID', 'SOURCE_KEY'])

df_ws2 = df_ws2.drop(columns = ['PLANT_ID', 'SOURCE_KEY'])

## We will now merge data for same plants: df_pg1 with df_ws1 and df_pg2 with df_ws2

df1 = pd.merge(df_pg1, df_ws1, on = 'DATE_TIME', how = 'left')

df2 = pd.merge(df_pg2, df_ws2, on = 'DATE_TIME', how = 'left')

## We will create a new column for DATE and TIME so that the code remains simple

df1['DATE'] = df1['DATE_TIME'].dt.date

df2['DATE'] = df2['DATE_TIME'].dt.date

df1['TIME'] = df1['DATE_TIME'].dt.time

df2['TIME'] = df2['DATE_TIME'].dt.time

## Filling in empty values with approximate assumptions

n = (27.862188+28.361993)/2

df1['IRRADIATION'] = df1['IRRADIATION'].fillna(n)



print('Station 1 value count-')

print('\tExpected values: ' + str(22*34*23*4))

print('\tValues received: ' + str(df1.shape[0]))

print('Station 2 value counts-')

print('\tExpected values: ' + str(22*34*23*4))

print('\tValues received: ' + str(df2.shape[0]))
tmp1 = df1.groupby('DATE', as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean', 'MODULE_TEMPERATURE': 'mean', 'AMBIENT_TEMPERATURE': 'mean', 'IRRADIATION': 'mean'})

tmp2 = df2.groupby('DATE', as_index = False).agg({'AC_POWER': 'mean', 'DC_POWER': 'mean', 'MODULE_TEMPERATURE': 'mean', 'AMBIENT_TEMPERATURE': 'mean', 'IRRADIATION': 'mean'})

fig = go.Figure()

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.DC_POWER/10-tmp1.AC_POWER, mode = 'lines', name = 'Station 1-Loss'))

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.DC_POWER-tmp2.AC_POWER, mode = 'lines', name = 'Station 2-Loss'))

fig.update_layout(title = 'Power Loss from Mean DC to Mean AC', xaxis_title = 'Dates', yaxis_title = 'Loss')

fig.show()



fig = make_subplots(rows = 1, cols = 2)

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.DC_POWER/10, mode = 'lines', name = 'Station 1-DC'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.DC_POWER, mode = 'lines', name = 'Station 2-DC'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.DC_POWER/10-tmp1.AC_POWER, mode = 'lines', name = 'Station 1-Loss'), row = 1, col = 2)

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.DC_POWER-tmp2.AC_POWER, mode = 'lines', name = 'Station 2-Loss'), row = 1, col = 2)

fig.update_layout(title = 'Comparison Plot between Power lost vs Power Generated', xaxis_title = 'Dates', yaxis_title = 'Power')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.AC_POWER, mode = 'lines', name = 'STATION 1'))

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.AC_POWER, mode = 'lines', name = 'STATION 2'))

fig.update_layout(title = 'AC Power Output Comparison Plot', xaxis_title = 'DATE/TIME', yaxis_title = 'Power Output')

fig.show()



fig = make_subplots(rows = 2, cols = 2)

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.IRRADIATION, mode = 'lines', name = 'IRR-ST1'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.IRRADIATION, mode = 'lines', name = 'IRR-ST2'), row = 1, col = 2)

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.MODULE_TEMPERATURE, mode = 'lines', name = 'MOD-ST1'), row = 2, col = 1)

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.MODULE_TEMPERATURE, mode = 'lines', name = 'MOD-ST2'), row = 2, col = 2)

fig.add_trace(go.Scatter(x = tmp1.DATE, y = tmp1.AMBIENT_TEMPERATURE, mode = 'lines', name = 'AMB-ST1'), row = 2, col = 1)

fig.add_trace(go.Scatter(x = tmp2.DATE, y = tmp2.AMBIENT_TEMPERATURE, mode = 'lines', name = 'AMB-ST2'), row = 2, col = 2)

fig.show()
tmp1 = df1.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

tmp2 = df1.groupby(['SOURCE_KEY', 'TIME'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig = make_subplots(rows = 1, cols = 2)

keys = df1['SOURCE_KEY'].unique()

for key in keys:

    fig.add_trace(go.Scatter(x = tmp1[tmp1['SOURCE_KEY'] == key].DATE, y = tmp1[tmp1['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key), row = 1, col = 1)

    fig.add_trace(go.Scatter(x = tmp2[tmp2['SOURCE_KEY'] == key].TIME, y = tmp2[tmp2['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key), row = 1, col = 2)

fig.update_layout(title = 'Station 1')

fig.show()



tmp1 = df2.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

tmp2 = df2.groupby(['SOURCE_KEY', 'TIME'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig = make_subplots(rows = 1, cols = 2)

keys = df2['SOURCE_KEY'].unique()

for key in keys:

    fig.add_trace(go.Scatter(x = tmp1[tmp1['SOURCE_KEY'] == key].DATE, y = tmp1[tmp1['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key), row = 1, col = 1)

    fig.add_trace(go.Scatter(x = tmp2[tmp2['SOURCE_KEY'] == key].TIME, y = tmp2[tmp2['SOURCE_KEY'] == key].AC_POWER, mode = 'lines', name = key), row = 1, col = 2)

fig.update_layout(title = 'Station 2')

fig.show()



tmp1 = df1.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

tmp2 = df2.groupby(['SOURCE_KEY', 'DATE'], as_index = False).agg({'DC_POWER': 'mean', 'AC_POWER': 'mean'})

fig = make_subplots(rows = 1, cols = 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

fig.add_trace(go.Pie(labels = tmp1['SOURCE_KEY'], values = tmp1['DC_POWER']/10, name = 'DC Production'), 1, 1)

fig.add_trace(go.Pie(labels = tmp1['SOURCE_KEY'], values = tmp1['AC_POWER'], name = 'AC Output'), 1, 2)

fig.update_traces(hoverinfo = 'label+percent+name')

fig.update_layout(title_text = 'Station 1: Pie Chart Comparison')

fig.show()

fig = make_subplots(rows = 1, cols = 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

fig.add_trace(go.Pie(labels = tmp2['SOURCE_KEY'], values = tmp2['DC_POWER'], name = 'DC Production'), 1, 1)

fig.add_trace(go.Pie(labels = tmp2['SOURCE_KEY'], values = tmp2['AC_POWER'], name = 'AC Output'), 1, 2)

fig.update_traces(hoverinfo = 'label+percent+name')

fig.update_layout(title_text = 'Station 2: Pie Chart Comparison')

fig.show()
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(df1[['IRRADIATION']], df1['DC_POWER']/10, test_size = 0.3, random_state = 0)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(df2[['IRRADIATION']], df2['DC_POWER'], test_size = 0.3, random_state = 0)

lin1 = LinearRegression()

lin2 = LinearRegression()

lin1.fit(X_train1, Y_train1)

lin2.fit(X_train2, Y_train2)

Y_pred1 = lin1.predict(X_test1)

Y_pred2 = lin2.predict(X_test2)

print('Train Accuracy')

print('\tModel 1 MAE: ', metrics.mean_absolute_error(Y_train1, lin1.predict(X_train1)))

print('\tModel 2 MAE: ', metrics.mean_absolute_error(Y_train2, lin2.predict(X_train2)))

print('Test Accuracy')

print('\tModel 1 MAE: ', metrics.mean_absolute_error(Y_test1, Y_pred1))

print('\tModel 2 MAE: ', metrics.mean_absolute_error(Y_test2, Y_pred2))
tmp1 = df1.groupby('DATE', as_index = False).agg({'IRRADIATION': 'mean'})

tmp2 = df2.groupby('DATE', as_index = False).agg({'IRRADIATION': 'mean'})

tmp1 = df1.rename(columns = {'DATE': 'ds', 'IRRADIATION': 'y'})

tmp2 = df2.rename(columns = {'DATE': 'ds', 'IRRADIATION': 'y'})

prop1 = fbprophet.Prophet(changepoint_prior_scale = 0.6)

prop2 = fbprophet.Prophet(changepoint_prior_scale = 0.6)

prop1.fit(tmp1)

prop2.fit(tmp2)

forecast1 = prop1.make_future_dataframe(periods = 15, freq = 'D')

forecast1 = prop1.predict(forecast1)

forecast2 = prop2.make_future_dataframe(periods = 15, freq = 'D')

forecast2 = prop2.predict(forecast2)
yhat1 = forecast1['yhat'].values.reshape(-1, 1)

yhat2 = forecast2['yhat'].values.reshape(-1, 1)

ypred1 = lin1.predict(yhat1)

ypred2 = lin2.predict(yhat2)



fig = go.Figure()

fig.add_trace(go.Scatter(x = forecast1['ds'], y = forecast1['yhat'], mode = 'lines', name = 'Station 1'))

fig.add_trace(go.Scatter(x = forecast2['ds'], y = forecast2['yhat'], mode = 'lines', name = 'Station 2'))

fig.update_layout(title = 'Forecast for next 15 days', xaxis_title = 'Date', yaxis_title = 'DC Production')

fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x = forecast1['ds'], y = ypred1, mode = 'lines', name = 'Station 1'))

fig.add_trace(go.Scatter(x = forecast2['ds'], y = ypred2, mode = 'lines', name = 'Station 2'))

fig.update_layout(title = 'Forecast for next 15 days', xaxis_title = 'Date', yaxis_title = 'DC Production')

fig.show()