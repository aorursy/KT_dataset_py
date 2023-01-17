import pandas as pd

import datetime

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.offline as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from fbprophet import Prophet



data = pd.read_csv("../input/indonesia-coronavirus-cases/confirmed_acc.csv")



data.tail()
end = datetime.datetime.now() - datetime.timedelta(1)

date_index = pd.date_range('2020-01-22', end)



fig = px.area(data, x=date_index, y='cases' )

fig.show()
df_prophet = data.rename(columns={"date": "ds", "cases": "y"})

df_prophet.tail()
from fbprophet.plot import plot_plotly

from fbprophet.plot import add_changepoints_to_plot



m = Prophet(

    changepoint_prior_scale=0.3, # increasing it will make the trend more flexible

    changepoint_range=0.99, # place potential changepoints in the first 95% of the time series

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



m.fit(df_prophet)



future = m.make_future_dataframe(periods=15)

forecast = m.predict(future)





forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
fig = plot_plotly(m, forecast)

py.iplot(fig) 



fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
forecast[50:70]