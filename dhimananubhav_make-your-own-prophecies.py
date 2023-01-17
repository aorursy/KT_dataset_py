%matplotlib inline



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from fbprophet import Prophet
df = pd.read_csv("../input/avocado.csv", parse_dates=["Date"])



del df['Unnamed: 0']

cols = ['Date', 'AveragePrice', 'type', 'region']

df = df[cols]

df = df[(df.region =='TotalUS') & (df.type == 'conventional') ] #& (df.Date >= '2016-01-01')



del df['region']

del df['type']



df = df.sort_values("Date")



df['Date'] = pd.to_datetime(df['Date'])

df = df.set_index('Date')



daily_df = df.resample('D').mean()

d_df = daily_df.reset_index().dropna()

d_df.columns = ['ds', 'y']



# Train test split 

n_weeks = 30

train = d_df[:-n_weeks]

test = d_df[-n_weeks:]
train.head()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



ax = sns.lineplot(x=train.ds, y=train.y, label="train")

ax = sns.lineplot(x=test.ds, y=test.y, label="test")



ax.axes.set_xlim(train.ds.min(), test.ds.max());
%%time

train['cap'] = train.y.max()

train['floor'] = train.y.min()



m = Prophet(growth='logistic', interval_width=0.8, changepoints=['2016-06-01'], changepoint_prior_scale=0.01)

m.add_seasonality(name='monthly', period=30.5, fourier_order=1)

m.add_seasonality(name='quarterly', period=91.25, fourier_order=5, prior_scale=0.1)

m.add_seasonality(name='yearly', period=365.25, fourier_order=10)



m.fit(train)
future = m.make_future_dataframe(periods=n_weeks, freq='W')

future['cap'] = 1.4 #approx max for last year same period

future['floor'] = train.y.min()



forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from  fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig2 = m.plot_components(forecast);
from datetime import datetime

horizon_days = (test.index[-1] - test.index[0])



from fbprophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(m, horizon= f'{horizon_days} days')

df_p = performance_metrics(df_cv)

df_p.head(5)
from fbprophet.plot import plot_cross_validation_metric

fig3 = plot_cross_validation_metric(df_cv, metric='mape')
test = pd.concat([test.set_index('ds'),forecast.set_index('ds')], axis=1, join='inner')



cols = ['y', 'yhat', 'yhat_lower', 'yhat_upper']

test = test[cols]

test['y'] = test.y

test['yhat'] = (test.yhat).round(2)

test['yhat_lower'] = (test.yhat_lower).round(2)

test['yhat_upper'] = (test.yhat_upper).round(2)



test.tail()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)



plt.plot(test.y)

plt.plot(test.yhat)

plt.legend();
test['e'] = test.y - test.yhat



rmse = np.sqrt(np.mean(test.e**2)).round(2)

mape = np.round(np.mean(np.abs(100*test.e/test.y)), 0)

print('RMSE = $', rmse)

print('MAPE =', mape, '%')
