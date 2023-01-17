import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews

from plotly.offline import iplot
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
# retrieve market and news dataframes
env = twosigmanews.make_env()
dfm, dfn = env.get_training_data()
print('market data shape:', dfm.shape)
dfm.head()
dfm.columns
dfm.isnull().sum()
dfm.assetCode.unique().shape[0]
days = dfm.groupby('assetCode').size()
days = pd.DataFrame(days, columns=['days']).reset_index().sort_values('days',ascending=False)

# plot barchart
data = [go.Bar(
            x=days.assetCode,
            y=days.days,
            name='No. of Days')]

layout = go.Layout(
    title='No. Companies with Days of Stock Data',
    yaxis=dict(
        title='No. of Days with Data'
    ),
    xaxis=dict(
        title='Company'
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)
day_count = pd.DataFrame(days['days'].value_counts()).reset_index()
day_count.columns = ['days','count']
day_count = day_count.sort_values('days',ascending=False)

# plot barchart
data = [go.Bar(
            x=day_count['days'],
            y=day_count['count'],
            name='No. of Days')]

layout = go.Layout(
    title='No. Companies with Days of Stock Data',
    yaxis=dict(
        title='No. of Companies'
    ),
    xaxis=dict(
        title='No. of Days with Stock Data'
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)
apple = dfm[dfm['assetCode']=='AAPL.O']

data1 = go.Scatter(
          x=apple.time,
          y=apple['close'],
          name='Price')

data2 = go.Bar(
            x=apple.time,
            y=apple.volume,
            name='Volume',
            yaxis='y2')

data = [data1, data2]

layout = go.Layout(
    title='Closing Price & Volume for AAPL.O',
    yaxis=dict(
        title='Price'
    ),
    yaxis2=dict(
        overlaying='y',
        side='right',
        range=[0, 1500000000], #increase upper range so that the volume bars are short
        showticklabels=False,
        showgrid=False
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)
data = [go.Candlestick(x=apple.time,
                       open=apple.open,
                       high=apple.open, #no data so used another column
                       low=apple.close, #no data so used another column
                       close=apple.close)]

# remove range slider
layout = go.Layout(
    xaxis = dict(
        rangeslider = dict(
            visible = False
        )
    )
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)
data = [go.Scatter(
          x=apple.time,
          y=apple['returnsOpenNextMktres10'],
          name='Price')]

iplot(data)
# for the target variable, change negative values into positive
dfm['abs_returnsOpenNextMktres10'] = dfm['returnsOpenNextMktres10'].apply(lambda x: abs(x))
# calculate the mean of absolute returnsOpenNextMktres10 for each company
# and treat that as a proxy for volatility
proxy_beta = dfm.groupby('assetCode', as_index=False)['abs_returnsOpenNextMktres10'].mean()


# plot barchart
data = [go.Bar(
            x=proxy_beta.assetCode,
            y=proxy_beta.abs_returnsOpenNextMktres10)]

layout = go.Layout(
    title='Volatile Companies',
    yaxis=dict(
        title='Proxy Beta'
    ),
    xaxis=dict(
        title='Company'
    )
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)