# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
print(df.head())
deleted_df = df.drop(columns=['time', 'test', 'negative'], inplace=False)
print(deleted_df)
import matplotlib.pyplot as plt

deleted_df.plot.line(subplots=True, label="confirmed", x='date', y="confirmed", color="blue", rot=45)
deleted_df.plot.line(subplots=True, label="released", x='date', y="released", color="green", rot=45)
deleted_df.plot.line(subplots=True, label="deceased", x='date', y="deceased", color="red", rot=45)
#deleted_df['released'].plot.line(x='date', label="released", color="green")
#deleted_df['deceased'].plot.line(x='date', label="deceased", color="red")

forecasting_df = pd.DataFrame({'ds': deleted_df['date'], 'y': df['confirmed']})
m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(forecasting_df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
print(forecast)
m.plot(forecast)
forecasting_df = pd.DataFrame({'ds': deleted_df['date'], 'y': df['released']})
m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(forecasting_df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
print(forecast)
m.plot(forecast)
forecasting_df = pd.DataFrame({'ds': deleted_df['date'], 'y': df['deceased']})
m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(forecasting_df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
print(forecast)
m.plot(forecast)