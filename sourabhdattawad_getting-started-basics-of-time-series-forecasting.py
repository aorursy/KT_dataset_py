# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

from fbprophet.plot import add_changepoints_to_plot

import plotly.offline as py

py.init_notebook_mode()

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from pandas import Series

from matplotlib import pyplot

from scipy.stats import boxcox

from pylab import rcParams

rcParams['figure.figsize'] = 18, 8
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

sales_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sales_df.head()
# Fetch values and time period

HOBBIES_1_001_CA_1 = sales_df[sales_df['id']=='HOBBIES_1_001_CA_1_validation']

TIME_PERIOD = HOBBIES_1_001_CA_1[HOBBIES_1_001_CA_1.columns[6:]].values[0]
# Plot the series

index = pd.date_range("29 01 2011", periods=1913,freq="d", name="date")

data = np.reshape(TIME_PERIOD,(-1,1))

wide_df = pd.DataFrame(data, index, ["HOBBIES_1_001_CA_1"])

plt.figure(figsize=(20, 6))

ax = sns.lineplot(data=wide_df)
HOBBIES_1_001_CA_1_df = pd.DataFrame({'y':HOBBIES_1_001_CA_1[HOBBIES_1_001_CA_1.columns[6:]].values[0]},index=index.values)

HOBBIES_1_001_CA_1_df.head()
index = pd.date_range("29 01 2011", periods=1913,freq="d", name="date")

rolling_mean = HOBBIES_1_001_CA_1_df.y.rolling(15).mean().values

HOBBIES_1_001_CA_1_df['mean_value'] = rolling_mean

data = np.reshape(rolling_mean,(-1,1))

wide_df = pd.DataFrame(data, index, ["HOBBIES_1_001_CA_1"])

plt.figure(figsize=(20, 6))

ax = sns.lineplot(data=wide_df)

plt.show()
pyplot.subplot(211)

pyplot.hist(rolling_mean)

pyplot.show()
#Applying 'Box-Cox transformation'

TRANSFORMED_DATA = HOBBIES_1_001_CA_1_df[HOBBIES_1_001_CA_1_df['mean_value'] > 0]

TRANSFORMED_DATA['shifted_value'],lam = boxcox(TRANSFORMED_DATA['mean_value'])

## Store the lambda value. This value is needed to inverse the transformed values.

print("Lamda value for box-cox transformation is ", lam)

pyplot.figure(1)

pyplot.subplot(211)

pyplot.plot(TRANSFORMED_DATA['shifted_value'])
pyplot.subplot(212)

pyplot.hist(TRANSFORMED_DATA['shifted_value'])

pyplot.show()
prophet_df = pd.DataFrame({'ds':TRANSFORMED_DATA.index,'y':TRANSFORMED_DATA['shifted_value']})

prophet_df.head()
m = Prophet()

m.fit(prophet_df)
future = m.make_future_dataframe(periods=28)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig2 = m.plot_components(forecast)
fig1 = m.plot(forecast)
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
EVENT_NAME = calendar_df[calendar_df.event_name_1.notnull()][['date','event_name_1']]

holidays = pd.DataFrame({'holiday':EVENT_NAME['event_name_1'],'ds':EVENT_NAME['date']})

holidays.head()
m = Prophet(changepoint_prior_scale=0.10,holidays=holidays)

m.fit(prophet_df)
forecast = m.predict(future)

fig2 = m.plot_components(forecast)
fig1 = m.plot(forecast)

fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)