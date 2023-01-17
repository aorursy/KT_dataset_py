#load necessary libraries
import pandas as pd
import numpy as np
from fbprophet import Prophet

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import the dataframes
# a1: PPW, a2: AB, a3: SC
df_a1 = pd.read_csv('/kaggle/input/workload-dataset/Area_A.csv')
df_a2 = pd.read_csv('/kaggle/input/workload-dataset/Area_B.csv')
df_a3 = pd.read_csv('/kaggle/input/workload-dataset/Area_C.csv')

# The following can be used to remove weekends from the dataframe(s)
# from pandas.tseries.offsets import BDay
#isBusinessDay = BDay().onOffset
#match_series = pd.to_datetime(df['Date']).map(isBusinessDay)
#df = df[match_series]

#Check datatypes 
df_a1.dtypes

# Convert 'Date' to datetime for processing
df_a1['Date'] = pd.to_datetime(df_a1['Date'])
df_a1.tail(10)
# Double-check 'Date' data type is correct
df_a1.dtypes
# Create new df for training model
df2_a1 = df_a1[['Date','Total']]                                
#inspect new df
df2_a1.tail(10)
df2_a1.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
#df2['ds'] = pd.to_datetime(df2['ds'])



# Train model on Area 1 data, and produce a 90 day forecast, taking MCMC samples to approximate posterior dist.
m1 = Prophet(mcmc_samples=100, seasonality_mode='multiplicative', interval_width=0.95)

# increasing max tree depth to avoid saturation
m1.fit(df2_a1, control={'max_treedepth':20})

future_a1 = m1.make_future_dataframe(periods=30)
future_df2_a1 = future_a1.copy() # for non-baseline predictions later on
future_a1.tail()
forecast_a1 = m1.predict(future_a1)

# Clip forecast to assign lower bound of zero, preventing negative workload forecast
forecast_a1['yhat_lower'] = forecast_a1['yhat_lower'].clip(lower = 0)
forecast_a1['yhat'] = forecast_a1['yhat'].clip(lower = 0)

# Check tail of forecast
forecast_a1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

df2_a1_forecast_plot = m1.plot(forecast_a1, xlabel = 'Date', ylabel = 'Area 1 Workload')
fig_a1_components = m1.plot_components(forecast_a1)
from fbprophet.diagnostics import cross_validation
df2_a1_cv = cross_validation(m1, initial='750 days', period='30 days', horizon = '365 days')
df2_a1_cv.head()
from fbprophet.diagnostics import performance_metrics
df_p1 = performance_metrics(df2_a1_cv)
df_p1.head()

df_a2.dtypes
df_a2['Date'] = pd.to_datetime(df_a2['Date'])
df_a2.dtypes
df2_a2 = df_a2[['Date','Total']]      
df2_a2.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
#df2['ds'] = pd.to_datetime(df2['ds'])

m2 = Prophet(mcmc_samples=100, seasonality_mode='multiplicative', interval_width=0.95)

m2.fit(df2_a2, control={'max_treedepth':20})

future_a2 = m2.make_future_dataframe(periods=30)
future_df2_a2 = future_a2.copy() # for non-baseline predictions later on
future_a2.tail()
forecast_a2 = m2.predict(future_a2)

forecast_a2['yhat_lower'] = forecast_a2['yhat_lower'].clip(lower = 0)
forecast_a2['yhat'] = forecast_a2['yhat'].clip(lower = 0)

forecast_a2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
df2_a2_forecast_plot = m2.plot(forecast_a2, xlabel = 'Date', ylabel = 'Area 2 Workload')
fig_a2_components = m2.plot_components(forecast_a2)
#from fbprophet.diagnostics import cross_validation
df2_a2_cv = cross_validation(m2, initial='365.25 days', period='30 days', horizon = '365 days')
df2_a2_cv.head()
#from fbprophet.diagnostics import performance_metrics
df_p2 = performance_metrics(df2_a2_cv)
df_p2.head()
df_a3.dtypes
df_a3['Date'] = pd.to_datetime(df_a3['Date'])
df_a3.dtypes
df_a3.head()
df2_a3 = df_a3[['Date','Total']]          
df2_a3.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
#df2_ab['ds'] = pd.to_datetime(df2_ab['ds'])

m3 = Prophet(mcmc_samples=100, seasonality_mode='additive', interval_width=0.95)

m3.fit(df2_a3, control={'max_treedepth':20})

future_a3 = m3.make_future_dataframe(periods=30)
future_df2_a3 = future_a3.copy() # for non-baseline predictions later on
future_a3.tail()
 
df2_a3.head(10)
forecast_a3 = m3.predict(future_a3)

forecast_a3['yhat_lower'] = forecast_a3['yhat_lower'].clip(lower = 0)
forecast_a3['yhat'] = forecast_a3['yhat'].clip(lower = 0)

forecast_a3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
df2_a3_forecast_plot = m3.plot(forecast_a3, xlabel = 'Date', ylabel = 'Area 3 Workload')
fig_a3_components = m3.plot_components(forecast_a3)
#from fbprophet.diagnostics import cross_validation
df2_a3_cv = cross_validation(m3, initial='365.25 days', period='30 days', horizon = '365 days')
df2_a3_cv.head()
#from fbprophet.diagnostics import performance_metrics
df2_p3 = performance_metrics(df2_a3_cv)
df2_p3.head()
#pip install plotly
%matplotlib inline

import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
trace = go.Scatter(
    name = 'Actual Workload',
    mode = 'markers',
    x = list(forecast_a1['ds']),
    y = list(df2_a1['y']),
    marker=dict(
        color='#6881CC',
        line=dict(width=1)
    )
)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(forecast_a1['ds']),
    y = list(forecast_a1['yhat']),
    marker=dict(
        color='#7B8296',
        line=dict(width=3)
    )
)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(forecast_a1['ds']),
    y = list(forecast_a1['yhat_upper']),
    line= dict(color='#7A9AFA'),
    fill = 'tonexty'
)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(forecast_a1['ds']),
    y = list(forecast_a1['yhat_lower']),
    line= dict(color='#7395FA')
)

tracex = go.Scatter(
    name = 'Actual Workload',
   mode = 'markers',
   x = list(df2_a1['ds']),
   y = list(df2_a1['y']),
   marker=dict(
      color='#7388C9',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Area 1 Workload',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure1=dict(data=data,layout=layout)

plt.savefig('Area 1.png')
#py.offline.iplot(figure1)
trace = go.Scatter(
    name = 'Actual Workload',
    mode = 'markers',
    x = list(forecast_a2['ds']),
    y = list(df2_a2['y']),
    marker=dict(
        color='#6881CC',
        line=dict(width=1)
    )
)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(forecast_a2['ds']),
    y = list(forecast_a2['yhat']),
    marker=dict(
        color='#7B8296',
        line=dict(width=3)
    )
)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(forecast_a2['ds']),
    y = list(forecast_a2['yhat_upper']),
    line= dict(color='#7A9AFA'),
    fill = 'tonexty'
)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(forecast_a2['ds']),
    y = list(forecast_a2['yhat_lower']),
    line= dict(color='#7395FA')
)

tracex = go.Scatter(
    name = 'Actual Workload',
   mode = 'markers',
   x = list(df2_a2['ds']),
   y = list(df2_a2['y']),
   marker=dict(
      color='#7388C9',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Area 2 Workload',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure2=dict(data=data,layout=layout)
plt.savefig('Area 2.png')
#py.offline.iplot(figure2)
trace = go.Scatter(
    name = 'Actual Workload',
    mode = 'markers',
    x = list(forecast_a3['ds']),
    y = list(df2_a3['y']),
    marker=dict(
        color='#6881CC',
        line=dict(width=1)
    )
)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(forecast_a3['ds']),
    y = list(forecast_a3['yhat']),
    marker=dict(
        color='#7B8296',
        line=dict(width=3)
    )
)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(forecast_a3['ds']),
    y = list(forecast_a3['yhat_upper']),
    line= dict(color='#7A9AFA'),
    fill = 'tonexty'
)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(forecast_a3['ds']),
    y = list(forecast_a3['yhat_lower']),
    line= dict(color='#7395FA')
)

tracex = go.Scatter(
    name = 'Actual Workload',
   mode = 'markers',
   x = list(df2_a3['ds']),
   y = list(df2_a3['y']),
   marker=dict(
      color='#7388C9',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Area 3 Workload',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure3=dict(data=data,layout=layout)
plt.savefig('Area 3.png')
# PLEASE NOTE: the chart for figure 3 is extremely "busy", and the Prophet generated

py.offline.iplot(figure1)
py.offline.iplot(figure2)
py.offline.iplot(figure3)
