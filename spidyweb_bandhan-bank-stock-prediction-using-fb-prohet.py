# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bandhan-bank-nse-data/BANDHANBNK.NS (2).csv')
data.head()
data.shape
data.isnull().any()
data.isnull().sum()
data.dropna(axis = 0, how ='any')
data = data.dropna(axis = 0, how ='any')
data.isnull().sum()
plt.figure(figsize=(14, 10))
plt.plot(data.index/15,data['Open'])
plt.title('Bandhan Bank Stock Price ')
plt.ylabel('Stock');
plt.xlabel('0-30 Months: from 27th March 2018 til 8th April 2020');
plt.show()
import fbprophet
# Prophet requires columns ds (Date) and y (value)
data = data.rename(columns={'Date': 'ds', 'Open': 'y'})

# Make the prophet model and fit on the data
data_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
data_prophet.fit(data)
# Make a future dataframe for 2 years or 90 days
data_forecast = data_prophet.make_future_dataframe(periods=90, freq='D')
# Make predictions
data_forecast = data_prophet.predict(data_forecast)
data_prophet.plot(data_forecast, xlabel = 'Date', ylabel = 'Stock')
plt.title('Stock Prediction of Bandhan Bank');
data_prophet.plot_components(data_forecast)
plt.title('Stock Prediction of Bandhan Bank');
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(data_prophet, data_forecast)  # This returns a plotly Figure
py.iplot(fig)