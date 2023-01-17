import pandas as pd
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
confirmedcases_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df1 = confirmedcases_df.groupby('Country/Region').sum().reset_index()
df1
k = df1[df1['Country/Region']=='India'].loc[:,'1/22/20':]
k
confirmedcases_India = k.values.tolist()[0]
confirmedcases_India
data=pd.DataFrame(columns=['ds','y'])
data
dates = list(confirmedcases_df.columns[4:])
dates = list(pd.to_datetime(dates))
dates
data['ds'] = dates
data['y'] = confirmedcases_India
data
prop= Prophet()
prop.fit(data)
future = prop.make_future_dataframe(periods=30)
prop_forecast = prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(30)
forecast
fig = plot_plotly(prop, prop_forecast)
fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')