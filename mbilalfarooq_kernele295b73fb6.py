# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df=pd.read_csv('../input/Output1.csv')
df1=pd.read_csv('../input/Output1.csv',usecols=[2,8])

# Any results you write to the current directory are saved as output.
df.head()
airwick=df1[3120:3380]
index=df.groupby(df['AnalysisName']).sum()
index
airwick
import matplotlib.pyplot as plt
plt.plot(airwick['Date'], airwick['Interest'])
plt.title('US Interest Price of Air Wack')
plt.ylabel('Price ($)');
plt.xticks(rotation='vertical')
plt.show()
small_aw=airwick[:10]
plt.plot(small_aw['Date'], small_aw['Interest'])
plt.title('US Interest Price of Air Wack')
plt.ylabel('Price ($)');
plt.xticks(rotation='vertical')
plt.show()

import fbprophet
# Prophet requires columns ds (Date) and y (value)
airwick = airwick.rename(columns={'Date': 'ds', 'Interest': 'y'})

# Make the prophet model and fit on the data
airwick_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
airwick_prophet.fit(airwick)
airwick_forecast = airwick_prophet.make_future_dataframe(periods=0 , freq='W')
# Make predictions
#airwick_forecast
airwick_forecast = airwick_prophet.predict(airwick_forecast)
airwick_prophet.plot(airwick_forecast, xlabel = 'Date', ylabel = 'Interest')
plt.title('AIRWICK Interest in USA ');
airwick_prophet.plot_components(airwick_forecast)
airwick_forecast
plt.figure(figsize=(10, 8))
plt.plot(airwick_forecast['ds'], airwick['y'], 'b-', label = 'Acutual')
plt.plot(airwick_forecast['ds'], airwick_forecast['yhat'], 'r-', label = 'Predicted')
plt.xlabel('Date'); plt.ylabel('Interest'); plt.title('Air Wick Interest')
plt.legend();
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
r2_score (airwick.y, airwick_forecast.yhat)
mean_squared_error(airwick.y, airwick_forecast.yhat)
mean_absolute_error(airwick.y, airwick_forecast.yhat)
airwick_forecast.to_csv('AirWickUsa.csv',sep=',')