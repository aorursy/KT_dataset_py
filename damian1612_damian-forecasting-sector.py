import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import random

import matplotlib.colors as mcolors

import datetime

from IPython.display import HTML



from fbprophet import Prophet

from fbprophet.diagnostics import cross_validation, performance_metrics

from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot, plot_plotly

import plotly.graph_objs as go



import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

import ipywidgets as widgets

init_notebook_mode(connected=True)



from datetime import datetime, date, timedelta



from scipy.integrate import odeint

df = pd.read_csv('../input/Book1.csv')
df.head()
df = df.drop(['id'], axis = 1)
df = pd.melt(

        df, 

        id_vars = ['Sector '], 

        var_name='Year',

        value_name='value'

    )

df.head(20)
df['Year'] = pd.to_datetime(df['Year'])

df.head()
sector = []



for m in df['Sector '].unique():

    temp = 'df_{}'.format(m.replace(" ", "").lower()) 

    sector.append(temp)

    vars()[temp] = df[df['Sector ']==m]
df_agriculture.head()
plt.figure(figsize=(8, 5));

plt.plot(df_agriculture.Year,df_agriculture.value );

plt.title('Value over the Years', size=15);

plt.xlabel('Year', size=15)

plt.ylabel('Value', size=15);

plt.show();
df_constructions.head()
plt.figure(figsize=(8, 5));

plt.plot(df_constructions.Year,df_constructions.value );

plt.title('Value over the Years', size=15);

plt.xlabel('Year', size=15)

plt.ylabel('Value', size=15);

plt.show();
df_electricitygasandwatersupply.head()
plt.figure(figsize=(8, 5));

plt.plot(df_electricitygasandwatersupply.Year,df_electricitygasandwatersupply.value );

plt.title('Value over the Years', size=15);

plt.xlabel('Year', size=15)

plt.ylabel('Value', size=15);

plt.show();
df_manufacturing.head()
plt.figure(figsize=(8, 5));

plt.plot(df_manufacturing.Year,df_manufacturing.value );

plt.title('Value over the Years', size=15);

plt.xlabel('Year', size=15)

plt.ylabel('Value', size=15);

plt.show();
df_mining.head()
plt.figure(figsize=(8, 5));

plt.plot(df_mining.Year,df_mining.value );

plt.title('Value over the Years', size=15);

plt.xlabel('Year', size=15)

plt.ylabel('Value', size=15);

plt.show();
df_prophet = df_mining[['Year','value']]

df_prophet.columns = ['ds','y']
m_nd = Prophet(

    changepoint_range=.8,

    changepoint_prior_scale=15,

    n_changepoints=3,

    yearly_seasonality=False,

    weekly_seasonality = False,

    daily_seasonality = False,

    seasonality_mode = 'additive')

m_nd.fit(df_prophet)

future_nd = m_nd.make_future_dataframe(periods=5, freq = 'Y')

fcst_no_daily = round(m_nd.predict(future_nd))
trace1 = {

  "fill": None, 

  "mode": "markers", 

  "name": "actual no. of Confirmed", 

  "type": "scatter", 

  "x": df_prophet.ds, 

  "y": df_prophet.y

}

trace2 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "upper_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_upper

}

trace3 = {

  "fill": "tonexty", 

  "line": {"color": "#57b8ff"}, 

  "mode": "lines", 

  "name": "lower_band", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat_lower

}

trace4 = {

  "line": {"color": "#eb0e0e"}, 

  "mode": "lines+markers", 

  "name": "prediction", 

  "type": "scatter", 

  "x": fcst_no_daily.ds, 

  "y": fcst_no_daily.yhat

}

data = [trace1, trace2, trace3, trace4]

layout = {

  "title": "Confirmed - Time Series Forecast", 

  "xaxis": {

    "title": "", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

  "yaxis": {

    "title": "Confirmed nCov - Hubei", 

    "ticklen": 5, 

    "gridcolor": "rgb(255, 255, 255)", 

    "gridwidth": 2, 

    "zerolinewidth": 1

  }, 

}

fig = go.Figure(data=data, layout=layout)

iplot(fig)