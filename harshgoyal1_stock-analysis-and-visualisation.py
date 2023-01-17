import plotly

plotly.tools.set_credentials_file(username='Harsh_Goyal', api_key='eLzL5wGCUsJzJy1j2eVI')
import quandl



quandl.ApiConfig.api_key='vVMHiQx-9Dg9uGutZzAP'



df = quandl.get('WIKI/MSFT')



df=df.reset_index()



df.head()
import plotly.plotly as py

import plotly.graph_objs as go



import pandas as pd

from datetime import datetime





trace = go.Ohlc(x=df['Date'],

                open=df['Open'],

                high=df['High'],

                low=df['Low'],

                close=df['Close'])

data = [trace]

py.iplot(data, filename='simple_candlestick')




layout = go.Layout(

    xaxis = dict(

        rangeslider = dict(

            visible = False

        )

    )

)



data = [trace]



fig = go.Figure(data=data,layout=layout)

py.iplot(fig, filename='simple_candlestick')




data = [trace]

layout = {

    'title': 'The Great Recession',

    'yaxis': {'title': 'Stock'},

    'shapes': [{

        'x0': '2016-12-09', 'x1': '2016-12-09',

        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',

        'line': {'color': 'rgb(30,30,30)', 'width': 1}

    }],

    'annotations': [{

        'x': '2016-12-09', 'y': 0.05, 'xref': 'x', 'yref': 'paper',

        'showarrow': False, 'xanchor': 'left',

        'text': 'Increase Period Begins'

    }]

}

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='aapl-recession-candlestick')


trace = go.Ohlc(x=df['Date'],

                open=df['Open'],

                high=df['High'],

                low=df['Low'],

                close=df['Close'],

                       increasing=dict(line=dict(color= '#17BECF')),

                       decreasing=dict(line=dict(color= '#7F7F7F')))

data = [trace]

py.iplot(data, filename='styled_candlestick')




open_data = [33.0, 33.3, 33.5, 33.0, 34.1]

high_data = [33.1, 33.3, 33.6, 33.2, 34.8]

low_data = [32.7, 32.7, 32.8, 32.6, 32.8]

close_data = [33.0, 32.9, 33.3, 33.1, 33.1]

dates = [datetime(year=2013, month=10, day=10),

         datetime(year=2013, month=11, day=10),

         datetime(year=2013, month=12, day=10),

         datetime(year=2014, month=1, day=10),

         datetime(year=2014, month=2, day=10)]



trace = go.Candlestick(x=dates,

                       open=open_data,

                       high=high_data,

                       low=low_data,

                       close=close_data)

data = [trace]

py.iplot(data, filename='candlestick_datetime')