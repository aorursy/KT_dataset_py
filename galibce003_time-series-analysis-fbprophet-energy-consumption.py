import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation, performance_metrics
df = pd.read_csv('../input/nyc-energy-consumption/nyc_energy_consumption.csv', parse_dates=True)
df.head()
df.shape
df.dtypes
df.isnull().sum()
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df.dtypes
df = df.set_index('timeStamp')
plt.figure(figsize = (12, 5))

plt.plot(df['demand'])
plt.show()
df = df.resample('D').mean()
df.shape
plt.figure(figsize = (12, 5))

plt.plot(df['demand'])
plt.show()
df_final = df.reset_index()[['timeStamp', 'demand']].rename({'timeStamp' : 'ds', 'demand' : 'y'}, axis = 'columns')
df_final.head()
train = df_final[(df_final['ds'] >= '2012-01-01') & (df_final['ds'] <= '2017-04-30')]
test = df_final[(df_final['ds'] > '2017-04-30')]
print('Lenth of train set : {}'.format(len(train)))
print('Lenth of test set : {}'.format(len(test)))
model = Prophet(interval_width = 0.95, yearly_seasonality = True)
model.fit(train)
future = model.make_future_dataframe(periods = 104)
future.head()
future.tail()
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.figure(figsize = (12, 5))

df_final.set_index('ds')['y'].plot(label = "Actual")
forecast.set_index('ds')['yhat'].plot(label = "Predicted")

plt.legend()
plt.show()
plt.figure(figsize = (12, 5))

model.plot(forecast)
plt.show()
model.plot_components(forecast)
plt.show()
plt.figure(figsize = (12, 5))

chagepoint_fig = model.plot(forecast)
add_changepoints_to_plot(chagepoint_fig.gca(), model, forecast)

plt.show()
model.changepoints
cv_results = cross_validation( model = model, initial = '731 days', horizon = '365 days')
df_p = performance_metrics(cv_results)
df_p