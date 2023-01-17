# essentials
import numpy as np 
import pandas as pd 

# visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# time series algorithm
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics

# reproducibility
np.random.seed(34)

# Jupyter magic
%reload_ext autoreload
%autoreload 2
%matplotlib inline

sns.set()
!ls              # list the file in the working directory
!pip list      # list the package version numbers for reproducibiity
df_shampoo_orig = pd.read_csv('../input/sales-of-shampoo-over-a-three-year-period/sales-of-shampoo-over-a-three-ye.csv', 
                              nrows=36,
                              skiprows = 1, 
                              names = ['ds', 'y'], 
                              parse_dates = True )
df = df_shampoo_orig
df
df.info()
df['ds'] = df.ds.apply(lambda x: "198"+x)
df.ds.head()
df['ds']=pd.to_datetime(df['ds'])+pd.tseries.offsets.MonthEnd(0)
df.head()
train = df[:24]
train.tail()
m = Prophet(weekly_seasonality=False, daily_seasonality=False, n_changepoints=2)
m.add_seasonality(name='yearly', period=12, fourier_order=5)
m.fit(train)
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
figure = m.plot(forecast)
fig_decompose = m.plot_components(forecast)
m2 = Prophet(weekly_seasonality=False, daily_seasonality=False, n_changepoints=2)
m2.add_seasonality(name='yearly', period=12, fourier_order=1)

m2.fit(train)
future2 = m2.make_future_dataframe(periods=12, freq='m')
forecast2 = m2.predict(future2)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig2 = m2.plot(forecast2)
fig2_decompose = m2.plot_components(forecast2)
forecast['cutoff'] = pd.to_datetime('1980-12-31')
forecast['y'] = df['y']
forecast.tail()
df_p = performance_metrics(forecast)
df_p.head()
forecast2['cutoff'] = pd.to_datetime('1980-12-31')
forecast2['y'] = df['y']
forecast2.tail()
df_p2 = performance_metrics(forecast2)
df_p2.head()
df_p.index = df_p['horizon']
df_p2.index = df_p2['horizon']

df_error_compare = df_p - df_p2
df_error_compare = df_error_compare.drop(columns=['horizon', 'coverage'])
df_error_compare.loc[:'365 days']
forecast_persist = forecast2.copy()
forecast_persist['cutoff'] = pd.to_datetime('1980-12-31')
forecast_persist['y'] = df['y']
forecast_persist['yhat'] = df.at[23,'y']
forecast_persist.tail()
df_persist = performance_metrics(forecast_persist)
df_persist.head()
df_persist.index = df_persist['horizon']

df_error_compare_persist = df_persist - df_p2
df_error_compare_persist = df_error_compare_persist.drop(columns=['horizon', 'coverage'])
df_error_compare_persist.loc[:'365 days']
df_air_orig = pd.read_csv('../input/internationalairlinepassengers/international-airline-passengers.csv', 
                              nrows=144,
                              skiprows = 1, 
                              names = ['ds', 'y'], 
                              parse_dates = True )
df_air = df_air_orig
df_air.head()
df_air.tail()
df_air.info()
df_air['ds']=pd.to_datetime(df_air['ds'])+pd.tseries.offsets.MonthEnd(0)
df_air.head()
df_air.tail()
df_air.info()
train = df_air[:120]
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=24, freq='m')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast)
fig_decompose = m.plot_components(forecast)
## Multiplicative seasonality
m2 = Prophet(seasonality_mode='multiplicative')
m2.fit(train)
future2 = m2.make_future_dataframe(periods=24, freq='m')
forecast = m2.predict(future2)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
fig = m.plot(forecast)
fig = m.plot_components(forecast)
forecast_persist = forecast.copy()
forecast_persist['cutoff'] = pd.to_datetime('1959-01')
forecast_persist['y'] = df_air['y']
forecast_persist['yhat'] = df_air.at[119,'y']
forecast_persist.tail()
fig = m.plot(forecast_persist)
forecast['cutoff'] = pd.to_datetime('1958-12-31')
forecast['y'] = df_air['y']
forecast.tail()
df_air_p = performance_metrics(forecast[120:])
df_air_p.head()
df_persist_p = performance_metrics(forecast_persist[120:])
df_persist_p.head()
df_air_plot = pd.DataFrame([df_air_p['rmse'], df_persist_p['rmse']])
df_air_plot = df_air_plot.T
df_air_plot.columns = ['prophet_rmse', 'persist_rmse']
df_air_plot.head()
df_plot = df_air_plot[:12]
df_plot
ax = sns.lineplot(
    data=df_plot,
    x=list(range(12)), 
    y='prophet_rmse',
    )
plt.title('RMSE Comparison of Prophet Model for Flight Passenger')
plt.xlabel('Month')
plt.ylabel('RMSE')
ax = sns.lineplot(
    data=df_plot,
    x=list(range(1, 13)), 
    y='prophet_rmse',
    )

ax = sns.lineplot(
    data=df_plot,
    x=list(range(1, 13)), 
    y='persist_rmse',
    )

plt.title('RMSE Comparison of Prophet Model for Flight Passenger')
plt.xlabel('Month')
plt.ylabel('RMSE')

plt.rcParams['figure.figsize']=(12, 6)
plt.legend(['Prophet RMSE','Persistence RMSE'])
df_air_compare = df_persist_p - df_air_p

df_air_compare = df_air_compare.drop(columns=['horizon', 'coverage'])
df_air_compare[:12]