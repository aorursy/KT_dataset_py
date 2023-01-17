import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

from scipy import stats

from fbprophet import Prophet

import logging

logging.getLogger().setLevel(logging.ERROR)



import matplotlib.pyplot as plt

%matplotlib inline



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/industrial-production-index-in-usa/INDPRO.csv')

df.head()
df.columns = ['Date', 'IPI']

df.head()
df['Date'] = pd.to_datetime(df['Date'])

df.set_index ('Date', inplace = True)

df.index
df_new = df['1998-01-01':]

df_new.head()
# to check NAs

df_new.info()

df_new.isnull().sum()
df_new.describe().transpose()
f, ax = plt.subplots(figsize = (16,10))

ax.plot(df_new, c = 'r');
df_new.columns
df_new.index
df_new = df_new.reset_index()

df_new.columns
df_new.columns = ['ds', 'y'] 

df_new.head()
#periods = 30

#train_df = df_new[:-periods]
m = Prophet()

m.fit(df_new)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)