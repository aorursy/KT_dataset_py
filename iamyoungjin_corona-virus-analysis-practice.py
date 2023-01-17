import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from datetime import date, timedelta

from statsmodels.tsa.arima_model import ARIMA

from sklearn.cluster import KMeans

from fbprophet import Prophet

trend = pd.read_csv('../input/coronavirusdataset/trend.csv',parse_dates=['date'])

route = pd.read_csv('../input/coronavirusdataset/route.csv',parse_dates=['date'])

patient = pd.read_csv('../input/coronavirusdataset/patient.csv',parse_dates=['confirmed_date'])

time = pd.read_csv('../input/coronavirusdataset/time.csv',parse_dates=['date'])
time.head()
plt.figure(figsize=(12,8))

plt.plot(time['date'],time['new_confirmed'],'ro:')

# sns.lineplot(data=time,x='date',y='new_confirmed',markers=True,dashes=True)

plt.xticks(rotation=30)

plt.title('How many COVID-19 confirmed peple')