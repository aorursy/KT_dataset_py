# - Libraries -

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import re

import datetime
# - Data -

url = '../input/netflix-stock-prices/netflix-stocks-data.csv'

data = pd.read_csv(url, header='infer')
# - Shape -

print("Total Records: ", data.shape[0])
#Check for missing/null values

data.isna().sum()
#Rename Close/Last Column

data.rename(columns={' Close/Last':'Close'}, inplace=True)
#Stripping Whitespace from the other Column Names

cols = data.columns

new_cols= []



#Strip White Space

for i in cols:

    new_cols.append(i.strip(' '))



#Rename Columns

for i, j in zip(cols, new_cols):

    data.rename(columns={i:j}, inplace=True)
#Remove the $ symbol

def extract(x):

    res = ''.join(re.findall(r"[0-9.0-9]",x))

    return res



col_flt = ['Close','Open','High', 'Low']  #Columns from which $ has to be removed



for i in col_flt:

    data[i] = data[i].apply(lambda x : extract(x))
#Inspect

data.head()
#Change Data Type

data.Date = data.Date.astype('datetime64[ns]')

data.Close = data.Close.astype('float')

data.Open = data.Open.astype('float')

data.High = data.High.astype('float')

data.Low = data.Low.astype('float')

# Candle Stick Charting

fig = go.Figure(data=[go.Candlestick(

    x=data['Date'],

    open=data['Open'], high=data['High'],

    low=data['Low'], close=data['Close'],

    increasing_line_color= 'cyan', decreasing_line_color= 'magenta'

)])



fig.update_layout(xaxis_rangeslider_visible=False)



fig.update_layout(

    title='Netflix Stock Prices (2010-2020)',

    yaxis_title='Stocks',

    shapes = [dict(

        x0='2018-01-21', x1='2018-01-21', y0=0, y1=1, xref='x', yref='paper',

        line_width=2),

              dict(

        x0='2013-01-23', x1='2013-01-21', y0=0, y1=1, xref='x', yref='paper',

        line_width=2)

             ],

    annotations=[dict(

        x='2018-01-21', y=0.04, xref='x', yref='paper',

        showarrow=False, xanchor='left', text='Drastic'),

                dict(

        x='2013-01-23', y=0.15, xref='x', yref='paper',

        showarrow=False, xanchor='left', text='Gradual')

                

                ]

)



fig.show()
#Creating a seperate dataframe with specific dates

mask = (data['Date'] >= '2020-01-01') & (data['Date'] <= '2020-07-17')

df = data.loc[mask]



fig = go.Figure(data=[go.Candlestick(

    x=df['Date'],

    open=df['Open'], high=df['High'],

    low=df['Low'], close=df['Close'],

    increasing_line_color= 'cyan', decreasing_line_color= 'gray'

)])



fig.update_layout(xaxis_rangeslider_visible=False)



fig.update_layout(

    title='Netflix Stock Prices (during Covid)',

    yaxis_title='Stocks',

    shapes = [dict(

        x0='2020-03-16', x1='2020-03-19', y0=0, y1=1, xref='x', yref='paper',

        line_width=2),

              dict(

        x0='2020-07-09', x1='2020-07-14', y0=0, y1=1, xref='x', yref='paper',

        line_width=2)],

    

    annotations=[dict(

        x='2020-03-20', y=0.5, xref='x', yref='paper',

        showarrow=False, xanchor='left', text='Lowest'),

                dict(

        x='2020-07-07', y=0.85, xref='x', yref='paper',

        showarrow=False, xanchor='right', text='Highest')                 

                

                ]

)



fig.show()