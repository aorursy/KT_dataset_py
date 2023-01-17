# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/my-prophet-ts-da/train_data_prophet.csv')

test = pd.read_csv('../input/my-prophet-ts-da/test_data_prophet.csv')
ax = train.set_index('ds').plot(figsize = (12, 4))

from fbprophet import Prophet
my_model = Prophet(interval_width = 0.8)

my_model.fit(train)

#We will try to do forecasting for first 12 weeks of 2019

future_dates = my_model.make_future_dataframe(periods = 12*7)

future_dates.tail()
forecast = my_model.predict(future_dates)

my_model.plot(forecast);
my_model.plot_components(forecast);
forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(84)
forecast_values.head(7)
# y hat series gives possible number of incidents per week (therefore modulo 7)

y_hat = forecast_values['yhat'].groupby((forecast_values.index) // 7).sum()

y_hat.reset_index().drop('index',axis=1)
#y hat upper gives upper limit of possible number of incidents per week

y_hat_upper = forecast_values['yhat_upper'].groupby((forecast_values.index) // 7).sum()

y_hat_upper.reset_index().drop('index',axis=1)
#Modulo 7 to get actual number of incidents per week 

test['y'].groupby(test.index // 7).sum()
plt.plot(y_hat.values)

plt.plot(y_hat_upper.values)

plt.plot(test['y'].groupby(test.index // 7).sum()[:12])