# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/weatherdata - Sheet1.csv")
from fbprophet import Prophet
df.head()
df['ds'] = df['Date']
df['y'] = df['Temperature']
df = df.drop(['Date', 'Temperature'], axis=1)
df.tail()
df['ds'] = pd.to_datetime(df['ds']).dt.date
df.tail()
import matplotlib.pyplot as plt
plt.plot(df['ds'], df['y'])
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)
m.plot_components(forecast)
from fbprophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(m, horizon='90 days')
df_p = performance_metrics(df_cv)
df_p.head()
from fbprophet.plot import plot_cross_validation_metric
plot_cross_validation_metric(df_cv, metric='mape')
