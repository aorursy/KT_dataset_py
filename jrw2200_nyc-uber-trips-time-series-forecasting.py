import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



import collections

import itertools



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



import scipy.stats as stats

from scipy.stats import norm

from scipy.special import boxcox1p



import statsmodels

import statsmodels.api as sm

#print(statsmodels.__version__)

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA



from fbprophet import Prophet



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        continue

        #print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv', parse_dates=['Date/Time'])

df = pd.concat([df,pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-may14.csv', parse_dates=['Date/Time'])], axis=0)

df = pd.concat([df,pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jun14.csv', parse_dates=['Date/Time'])], axis=0)

df = pd.concat([df,pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv', parse_dates=['Date/Time'])], axis=0)

df = pd.concat([df,pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv', parse_dates=['Date/Time'])], axis=0)

df = pd.concat([df,pd.read_csv('/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv', parse_dates=['Date/Time'])], axis=0)

df = df.rename(columns={'Date/Time': 'Datetime'})

print(df.shape)
df.dtypes
df_H  = pd.DataFrame()

df_H['raw'] = df.Datetime.value_counts().resample('H', how='sum')

df_H['log'] = np.log(df_H.raw)

df_H['raw_diff'] = df_H['raw'].diff()

df_H['log_diff'] = df_H['log'].diff()

df_H.head(20)
fig = go.Figure()



fig.add_trace(go.Scatter(

                x = df_H.index,

                y=df_H.raw,

                name="No. trips",

                mode='lines+markers',

                line_color='blue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x = df_H.index,

                y=df_H.raw_diff,

                name="Difference [No. trips]",

                mode='lines+markers',

                line_color='red',

                opacity=0.8))



# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['2014-04-01','2014-04-15'],

#                  title_text="Manually Set Date Range")



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

                x = df_H.index,

                y=df_H.log,

                name="log(No. trips)",

                mode='lines+markers',

                line_color='blue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x = df_H.index,

                y=df_H.log_diff,

                name="Difference[ log(No. trips)]",

                mode='lines+markers',

                line_color='red',

                opacity=0.8))



# Use date string to set xaxis range

#fig.update_layout(xaxis_range=['2014-04-01','2014-04-15'],

                  #title_text="Manually Set Date Range")



fig.show()
adf = adfuller(df_H["raw"])

print("p-value of raw series: {}".format(float(adf[1])))



adf = adfuller(df_H["raw_diff"].dropna())

print("p-value of raw series: {}".format(float(adf[1])))



adf = adfuller(df_H["log"])

print("p-value of raw series: {}".format(float(adf[1])))



adf = adfuller(df_H["log_diff"].dropna())

print("p-value of raw series: {}".format(float(adf[1])))
# Acknowledgements