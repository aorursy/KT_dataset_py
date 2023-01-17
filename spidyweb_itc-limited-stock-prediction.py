# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/itc-limitednse-stock/ITC.NS (1).csv')

data.head()
data.shape
data.isnull().any()
data.isnull().sum()
data = data.dropna(axis = 0, how ='any')
data.isnull().sum()
data.shape
import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 10))
plt.plot(data.index/365,data['Open'])
plt.title('ITC Opening Stock Price(in Rupee)')
plt.ylabel('Opening Stock');
plt.xlabel('from 1st January 1996 to 8th July 2020');
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
plt.title('Opeing Stock Prediction of ITC Limited');
data_prophet.plot_components(data_forecast)
plt.title('Opeing Stock Prediction of ITC Limited');
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(data_prophet, data_forecast)  # This returns a plotly Figure
py.iplot(fig)