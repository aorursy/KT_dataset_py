import pandas as pd

from fbprophet import Prophet
df = pd.read_csv("../input/example-wp-log-peyton-manningcsv/example_wp_log_peyton_manning.csv")

df.head(5)
m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2  = m.plot_components(forecast)
df = pd.read_csv("../input/example-wp-log-r/wp_log_R.csv")

df.head(5)
df['cap'] = 8.5
m = Prophet(growth='logistic')

m.fit(df)
future = m.make_future_dataframe(periods=1826)

future['cap'] = 8.5

fcst = m.predict(future)

fig = m.plot(fcst)
df['y'] = 10 - df['y']

df['cap'] = 6

df['floor'] = 1.5

future['cap'] = 6

future['floor'] = 1.5

m = Prophet(growth='logistic')

m.fit(df)

fcst = m.predict(future)

fig = m.plot(fcst)
from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
m = Prophet(changepoint_prior_scale=0.5)

forecast = m.fit(df).predict(future)

fig = m.plot(forecast)
#checking prior scale at a very low value

m = Prophet(changepoint_prior_scale=0.001)

forecast = m.fit(df).predict(future)

fig = m.plot(forecast)
m = Prophet(changepoints=['2012-04-01'])

forecast = m.fit(df).predict(future)

fig = m.plot(forecast)
playoffs = pd.DataFrame({

  'holiday': 'playoff',

  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',

                        '2010-01-24', '2010-02-07', '2011-01-08',

                        '2013-01-12', '2014-01-12', '2014-01-19',

                        '2014-02-02', '2015-01-11', '2016-01-17',

                        '2016-01-24', '2016-02-07']),

  'lower_window': 0,

  'upper_window': 1,

})

superbowls = pd.DataFrame({

  'holiday': 'superbowl',

  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),

  'lower_window': 0,

  'upper_window': 1,

})

holidays = pd.concat((playoffs, superbowls))
m = Prophet(holidays=holidays)

forecast = m.fit(df).predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][

        ['ds', 'playoff', 'superbowl']][-10:]
fig = m.plot_components(forecast)
m = Prophet(holidays=holidays)

m.add_country_holidays(country_name='US')

m.fit(df)
m.train_holiday_names
forecast = m.predict(future)

fig = m.plot_components(forecast)
from fbprophet.plot import plot_yearly

m = Prophet().fit(df)

a = plot_yearly(m)
from fbprophet.plot import plot_yearly

m = Prophet(yearly_seasonality=20).fit(df)

a = plot_yearly(m)
m = Prophet(weekly_seasonality=False)

m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

forecast = m.fit(df).predict(future)

fig = m.plot_components(forecast)
def is_nfl_season(ds):

    date = pd.to_datetime(ds)

    return (date.month > 8 or date.month < 2)



df['on_season'] = df['ds'].apply(is_nfl_season)

df['off_season'] = ~df['ds'].apply(is_nfl_season)
m = Prophet(weekly_seasonality=False)

m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')

m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')



future['on_season'] = future['ds'].apply(is_nfl_season)

future['off_season'] = ~future['ds'].apply(is_nfl_season)

forecast = m.fit(df).predict(future)

fig = m.plot_components(forecast)
df = pd.read_csv("../input/air-passenger/air_passengers.csv")

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(50, freq='MS')

forecast = m.predict(future)

fig = m.plot(forecast)
m = Prophet(seasonality_mode='multiplicative')

m.fit(df)

forecast = m.predict(future)

fig = m.plot(forecast)
fig = m.plot_components(forecast)
m = Prophet(seasonality_mode='multiplicative')

m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')

m.add_regressor('regressor', mode='additive')
forecast = Prophet(interval_width=0.95).fit(df).predict(future)
m = Prophet(mcmc_samples=300)

forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
df = pd.read_csv("../input/yosemite-tempscsv/yosemite_temps.csv")

m = Prophet(changepoint_prior_scale=0.01).fit(df)

future = m.make_future_dataframe(periods=300, freq='H')

fcst = m.predict(future)

fig = m.plot(fcst)
fig = m.plot_components(fcst)
df2 = df.copy()

df2['ds'] = pd.to_datetime(df2['ds'])

df2 = df2[df2['ds'].dt.hour < 6]

m = Prophet().fit(df2)

future = m.make_future_dataframe(periods=300, freq='H')

fcst = m.predict(future)

fig = m.plot(fcst)
future2 = future.copy()

future2 = future2[future2['ds'].dt.hour < 6]

fcst = m.predict(future2)

fig = m.plot(fcst)
df = pd.read_csv("../input/retail-sales/retail_sales.csv")

m = Prophet(seasonality_mode='multiplicative').fit(df)

future = m.make_future_dataframe(periods=3652)

fcst = m.predict(future)

fig = m.plot(fcst)
m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(df)

fcst = m.predict(future)

fig = m.plot_components(fcst)
future = m.make_future_dataframe(periods=120, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)