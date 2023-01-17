# Check files

!ls ../input/*
# Set up modules/libraries

import pandas as pd

import numpy as np

from pandasql import sqldf

sql = lambda q: sqldf(q, globals())



# Data Viz libraries

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots

import plotly.graph_objects as go



# Config plot style sheets

# https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html

plt.style.use('fivethirtyeight')

pd.plotting.register_matplotlib_converters()



# suppress warnings

import warnings

warnings.filterwarnings("ignore")
#import BOM dataset from Kaggle

weather_AU = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

weather_AU.head(5)
weather_AU.describe().transpose()
print('min', weather_AU['Date'].min())

print('max', weather_AU['Date'].max())
# Import Australian BOM weather observation data

# Ingest daily maximum temperature for Sydney, Brisbane and Melbourne

weather_brisbane = pd.read_csv('../input/bom-weather-observation-data-select-stations/IDCJAC0010_040913_1800_Data.csv') # 040913 is Brisbane



weather_brisbane.head(10)
# Let's see spread of data

weather_brisbane.describe().transpose()
# See how many missing values

weather_brisbane[weather_brisbane.isna().any(axis=1)]
# Let's get rid of the missing values - get rid of the row if any NaNs

weather_brisbane.drop(columns=['Bureau of Meteorology station number', 'Product code', 'Days of accumulation of maximum temperature', 'Quality'], inplace=True)



weather_brisbane.dropna(axis=0, how='any', inplace=True)



weather_brisbane.head(10)
weather_brisbane['Date'] = pd.to_datetime(weather_brisbane[['Year', 'Month', 'Day']])



weather_brisbane.drop(columns=['Year', 'Month', 'Day'], inplace=True)



weather_brisbane.head(10)
# Let's set index to be datetime so we can filter easily

weather_brisbane.set_index('Date', inplace=True)
weather_brisbane.dtypes
# Given forecast weather, let's have a look at autocorrelation at the averaged to the monthly level

from pandas.plotting import autocorrelation_plot



weather_brisbane_monthly = weather_brisbane.resample('M').mean()



autocorrelation_plot(weather_brisbane_monthly['Maximum temperature (Degree C)'])
# Show interactive plot limited to date range

fig = go.Figure()

fig.add_trace(go.Scatter(x=weather_brisbane.index

                         ,y=weather_brisbane['Maximum temperature (Degree C)']

                         ,name='Weather - Brisbane observation for Max Temp (c)'

                         ,line_color='deepskyblue'

                         )

             )

fig.update_layout(title_text='Interactive - Brisbane weather max temperature'

                  ,xaxis_range=['1999-01-01','2019-12-31']

                  ,xaxis_rangeslider_visible=True)

fig.show()
# More comprehensive profiling



# 3 aspects of EDA that it captures:

# 1. Data Quality - ie df.dtypes and df.describe

# 2. Variable relationship - Pearson correlation - sns.heatmap(df.corr().annot=True)

# 3. Data spread - mean, std dev, median, min, max, histograms - sns.boxplot('variable', data=df)



import pandas_profiling

weather_brisbane.profile_report(style={'full_width':True})
print("min", weather_brisbane.index.min())

print("max", weather_brisbane.index.max())
train = weather_brisbane



train.reset_index(inplace=True) # Reset index

train.rename(columns={'Date': 'ds', 'Maximum temperature (Degree C)': 'y'}, inplace=True)



train.head(10)
# Create Prophet model - and fit training data to the model

# We set changepoint range to 80% and MCMC sampling to 100 - MCMC sampling adds uncertainty interval to seasonality

from fbprophet import Prophet

model = Prophet(changepoint_range=0.8, mcmc_samples=100)



model.fit(train)
# Using helper, create next forecast range to be next 2 years (720 days) - 

# aggregated to daily basis (since we only have daily readings)

from fbprophet import Prophet

future = model.make_future_dataframe(periods=720, freq='D')



future.tail()
# Make prediction using - show confidence interval 

# By default, Prophet uses Monte Carlo Markov Chain sampling to create confidence interval - 

# which covers 80% (not 95%) of the samples

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# Plot forecast

# Plot interactive plotly viz

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(model, forecast)  # This returns a plotly Figure



fig.update_layout(title_text='Interactive - Brisbane weather max temperature with forecast')



py.iplot(fig)
# Plot components of forecast in detail

model.plot_components(forecast)
# Cross validation between actuals and forecasted data between cutoff and horizon

from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(model, period='180 days', horizon ='365 days')

df_cv.head()
# See diagnostics of Prophet

from fbprophet.diagnostics import performance_metrics

df_perf = performance_metrics(df_cv)

df_perf.describe()
# mean absolute percentage error (MAPE) visualised

from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')