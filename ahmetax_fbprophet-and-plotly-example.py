%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

estimated_days=91
names = ['Date','Price(USD)']
df = pd.read_csv('../input/btcusd.csv',names=names)
df = df[1:]     # remove the first row
df.head()
df = df[['Date','Price(USD)']]


df = df.rename(columns={'Date': 'ds', 'Price(USD)': 'y'})

df.head()
df.info()
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
df['y'] = pd.to_numeric(df['y'],errors='ignore')
df = df[df['ds']>='2014-01-01']
df0=df.copy()
df = df[:-estimated_days]
# Sort records by ds if not sorted
# df = df.sort_values(by=['ds'],ascending=True)
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
df_forecast = df_prophet.predict(df_forecast)
# plot_components() draws 4 graps showing:
#     - trend line
#     - yearly seasonality
#     - weekly seasonality
#     - daily seasonality
df_prophet.plot_components(df_forecast)

# Draw forecast results
df_prophet.plot(df_forecast, xlabel = 'Date', ylabel = 'Bitcoin Price (USD)')

# Combine all graps in the same page
plt.title(f'{estimated_days} daily BTC/USD Estimation')
plt.title('BTC/USD Price')
plt.ylabel('BTC (USD)')
plt.show()

trace = go.Scatter(
    name = 'Actual price',
    mode = 'markers',
    x = list(df_forecast['ds']),
    y = list(df['y']),
    marker=dict(
        color='#FFBAD2',
        line=dict(width=1)
    )
)

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)

upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)

lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)

tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df0['ds']),
   y = list(df0['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)

data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)

plt.savefig('btc02.png')


py.offline.iplot(figure)
# plt.show()
df = df0.copy()
df = df[df['ds']>='2016-01-01']
df1=df.copy()
df = df[:-estimated_days]
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
df_forecast = df_prophet.predict(df_forecast)
trace = go.Scatter(
    name = 'Actual price',
    mode = 'markers',
    x = list(df_forecast['ds']),
    y = list(df['y']),
    marker=dict(
        color='#FFBAD2',
        line=dict(width=1)
    )
)
trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)
upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)
lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)
tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df1['ds']),
   y = list(df1['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)
plt.savefig('btc03.png')
py.offline.iplot(figure)

df = df1.copy()
df = df[df['ds']>='2017-01-01']
df2=df.copy()
df = df[:-estimated_days]
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
df_forecast = df_prophet.predict(df_forecast)
trace = go.Scatter(
    name = 'Actual price',
    mode = 'markers',
    x = list(df_forecast['ds']),
    y = list(df['y']),
    marker=dict(
        color='#FFBAD2',
        line=dict(width=1)
    )
)
trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)
upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)
lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)
tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df2['ds']),
   y = list(df2['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)
plt.savefig('btc04.png')
py.offline.iplot(figure)

