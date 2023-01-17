import pandas as pd 

import plotly.graph_objects as go

import plotly.offline as py

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

py.init_notebook_mode()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/coronavirusdataset/time.csv')

df_prophet = df.rename(columns={

    'date' : 'ds',

    'acc_confirmed' : 'y'

})
m = Prophet(

    changepoint_prior_scale=0.2,

    changepoint_range=0.98,

    yearly_seasonality=False,

    weekly_seasonality=False,

    daily_seasonality=True,

    seasonality_mode='additive'

)



m.fit(df_prophet)



future = m.make_future_dataframe(periods=7)

forecast = m.predict(future)



fig = plot_plotly(m, forecast)

py.iplot(fig)
fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)