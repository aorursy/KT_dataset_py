# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import iplot

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv")
data.head()
data.describe().T
# data shape

data.shape
# data columns

data.columns
# data types

data.dtypes
data.isnull().sum()
f,ax = plt.subplots(figsize = (12,7))

plt.subplot(2,1,1) 

sns.distplot(data.Open,color="green",label="Open Price");

plt.title("Open Price",fontsize = 20,color='blue')

plt.xlabel('Price',fontsize = 15,color='blue')

plt.legend()

plt.grid()

#

plt.subplot(2,1,2)

sns.distplot(data.Close,color="darkblue",label="Close Price");

plt.title("Close Price",fontsize = 20,color='blue')

plt.xlabel('Price',fontsize = 15,color='blue')

plt.tight_layout()

plt.legend()

plt.grid()
# Creating trace1

line_1 = go.Scatter(

                    x = data.index,

                    y = data.Open,

                    mode = "lines",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))  

data_line = [line_1]

layout = dict(title = 'TESLA Stock Price',

              xaxis= dict(title= 'Open Price',ticklen= 5,zeroline= False)

             )

fig = dict(data = data_line, layout = layout)

iplot(fig)
f,ax = plt.subplots(figsize = (10,7))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax);
# load library

from fbprophet import Prophet
data.head()
# convert date to date:)

data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)
prophet_df = data.iloc[:,[0,1]]

prophet_df.head()
prophet_df = prophet_df.rename(columns={'Date':'ds', 'Open':'y'})

prophet_df.tail(10)
prophet_df.dtypes
# Create Model

m = Prophet()

m.fit(prophet_df)
# Forcasting into the future

future = m.make_future_dataframe(periods=730)

future.tail(10)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# You can plot the forecast

figure1 = m.plot(forecast, xlabel='Date', ylabel='Price')
# If you want to see the forecast components

figure2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)