import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from fbprophet import Prophet

import statsmodels.api as sm

from scipy import stats

from plotly import tools

import chart_studio.plotly as py

import plotly.figure_factory as ff

import plotly.tools as tls

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")



plt.style.use("seaborn-whitegrid")
df = pd.read_csv('/kaggle/input/lt-finance-holdings-ltd-stock-price-2017-to-2020/LT_Finance_Holdings_Ltd_Stock_Price_2017_to_2020.csv',parse_dates=['Date'])
df.head()
# Brief Description of our dataset

df.describe()
Lt_df = df.copy()

# Change to datetime datatype.

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

ax1.plot(Lt_df["Date"], Lt_df["Close Price"])

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price")

ax1.set_title(" L&T Close Price History")



# Second Subplot



plt.figure(1)

ax1.plot(Lt_df["Date"], Lt_df["High Price"], color="green")

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price")

ax1.set_title(" L&T High Price History")



# Third Subplot

ax1.plot(Lt_df["Date"], Lt_df["Low Price"], color="red")

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Stock Price")

ax1.set_title(" L&T Low Price History")





# Fourth Subplot

ax2.plot(Lt_df["Date"], Lt_df["WAP"], color="orange")

ax2.set_xlabel("Date", fontsize=12)

ax2.set_ylabel("Stock Price")

ax2.set_title("Amazon's Volume History")

plt.show()
m = Prophet()



# Drop the columns

ph_df = Lt_df.drop(['Open Price', 'High Price', 'Low Price','WAP'], axis=1)

ph_df.rename(columns={'Close Price': 'y', 'Date': 'ds'}, inplace=True)



ph_df.head()
m = Prophet()



m.fit(ph_df)
future_prices = m.make_future_dataframe(periods=365)



# Predict Prices

forecast = m.predict(future_prices)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import matplotlib.dates as mdates



# Dates

starting_date = dt.datetime(2018, 4, 7)

starting_date1 = mdates.date2num(starting_date)

trend_date = dt.datetime(2018, 6, 7)

trend_date1 = mdates.date2num(trend_date)



pointing_arrow = dt.datetime(2018, 2, 18)

pointing_arrow1 = mdates.date2num(pointing_arrow)



# Learn more Prophet tomorrow and plot the forecast for amazon.

fig = m.plot(forecast)

ax1 = fig.add_subplot(111)

ax1.set_title("L&T Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Close Price", fontsize=12)



# Forecast initialization arrow

ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 1350), xytext=(starting_date1,1700),

            arrowprops=dict(facecolor='#ff7f50', shrink=0.1),

            )



# Trend emphasis arrow

ax1.annotate('Upward Trend', xy=(trend_date1, 1225), xytext=(trend_date1,950),

            arrowprops=dict(facecolor='#6cff6c', shrink=0.1),

            )



ax1.axhline(y=1260, color='b', linestyle='-')
fig2 = m.plot_components(forecast)

plt.show()

m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

future = m.make_future_dataframe(periods=12, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)

plt.title("Monthly Prediction \n 1 year time frame")



plt.show()
fig = m.plot_components(fcst)

plt.show()

trace = go.Ohlc(x=Lt_df['Date'],

                open=Lt_df['Open Price'],

                high=Lt_df['High Price'],

                low=Lt_df['Low Price'],

                close=Lt_df['Close Price'],

               increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = {

    'title': 'L&T Historical Price',

    'xaxis': {'title': 'Date',

             'rangeslider': {'visible': False}},

    'yaxis': {'title': 'Stock Price (USD$)'},

    'shapes': [{

        'x0': '2016-12-09', 'x1': '2016-12-09',

        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',

        'line': {'color': 'rgb(30,30,30)', 'width': 1}

    }],

    'annotations': [{

        'x': '2017-01-20', 'y': 0.05, 'xref': 'x', 'yref': 'paper',

        'showarrow': False, 'xanchor': 'left',

        'text': 'President Donald Trump <br> takes Office'

    }]

}



data = [trace]



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple_ohlc')
last_two = Lt_df.loc[Lt_df['Date'].dt.year > 2016]



g = last_two.groupby(["Date"])

monthly_averages = g.aggregate({"Open Price": np.mean, "High Price": np.mean, "Low Price": np.mean, "Close Price":np.mean})

monthly_averages.reset_index(level=0, inplace=True)



trace = go.Candlestick(x=monthly_averages['Date'],

                       open=monthly_averages['Open Price'].values.tolist(),

                       high=monthly_averages['High Price'].values.tolist(),

                       low=monthly_averages['Low Price'].values.tolist(),

                       close=monthly_averages['Close Price'].values.tolist(),

                      increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = {

    'title': 'L&T Historical Price <br> <i>For the Last two years </i>',

    'xaxis': {'title': 'Date',

             'rangeslider': {'visible': False}},

    'yaxis': {'title': 'Stock Price (USD$)'},

    'shapes': [{

        'x0': '2018-01-02', 'x1': '2018-01-02',

        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',

        'line': {'color': 'rgb(30,30,30)', 'width': 1}

    }],

    'annotations': [{

        'x': '2018-01-07', 'y': 0.9, 'xref': 'x', 'yref': 'paper',

        'showarrow': True, 'xanchor': 'left',

        'text': 'Upward Trend'

    }]

}

data = [trace]



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple_ohlc')
Lt_df['month_year'] = pd.to_datetime(Lt_df['Date']).dt.to_period('M')



# 2017 onwards

last_year = Lt_df.loc[Lt_df['Date'].dt.year > 2017]

g = last_year.groupby(["Date"])

monthly_averages = g.aggregate({"Open Price": np.mean, "High Price": np.mean, "Low Price": np.mean, "Close Price":np.mean})

monthly_averages.reset_index(level=0, inplace=True)



monthly_averages.dtypes





trace = go.Candlestick(x=monthly_averages['Date'],

                       open=monthly_averages['Open Price'].values.tolist(),

                       high=monthly_averages['High Price'].values.tolist(),

                       low=monthly_averages['Low Price'].values.tolist(),

                       close=monthly_averages['Close Price'].values.tolist(),

                      increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))





layout = {

    'title': 'L&T Historical Price <br> <i>A closer look to the upward trend </i>',

    'xaxis': {'title': 'Date',

             'rangeslider': {'visible': False}},

    'yaxis': {'title': 'Stock Price (USD$)'}

}





data = [trace]



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple_ohlc')



Lt_df.head()
Lt_df['10_d_avg'] = Lt_df['Close Price'].rolling(window=10).mean()

Lt_df['50_d_avg'] = Lt_df['Close Price'].rolling(window=50).mean()

Lt_df['200_d_avg'] = Lt_df['Close Price'].rolling(window=200).mean()

close_p = Lt_df['Close Price'].values.tolist()





# Variables to insert into plotly

ten_d = Lt_df['10_d_avg'].values.tolist()

fifty_d = Lt_df['50_d_avg'].values.tolist()

twoh_d = Lt_df['200_d_avg'].values.tolist()

date = Lt_df['Date'].values.tolist()



# Set date as index

amzn_df = Lt_df.set_index('Date')

fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True)



colors = ['#ff4500', '#92a1cf', '#6E6E6E']

avgs = ['10_d_avg', '50_d_avg', '200_d_avg']

# for i,c in zip(range(n),color):

#    ax1.plot(x, y,c=c)



for col, c in zip(avgs, colors):

    fig.append_trace({'x': Lt_df.index, 'y': Lt_df[col], 'type': 'scatter', 'name': col, 'line': {'color': c}}, 1, 1)

for col in ['Close Price']:

    fig.append_trace({'x': amzn_df.index, 'y': amzn_df[col], 'type': 'scatter', 'name': 'Closing Price', 'line':{'color': '#01DF3A'}}, 2, 1)

    

fig['layout'].update(height=800,title='Relationship between MAs <br> and Closing Price',

                    paper_bgcolor='#F2DFCE', plot_bgcolor='#F2DFCE')

    

iplot(fig, filename='pandas/mixed-type subplots')
Lt_df = Lt_df.reset_index()



# Plotly

trace0 = go.Scatter(

    x = Lt_df['Date'],

    y = ten_d,

    name = '10-day MA',

    line = dict(

        color = ('#ff6347'),

        width = 4)

)

trace1 = go.Scatter(

    x = Lt_df['Date'],

    y = fifty_d,

    name = '50-day MA',

    line = dict(

        color = ('#92a1cf'),

        width = 4,

    dash="dot")

)

trace2 = go.Scatter(

    x = Lt_df['Date'],

    y = twoh_d,

    name = '200-day MA',

    line = dict(

        color = ('#2EF688'),

        width = 4,

        dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'

)



data = [trace0, trace1, trace2]





# Edit the layout

layout = dict(title = 'Moving Averages for L&T',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Price'),

             

              paper_bgcolor='#FFF9F5',

              plot_bgcolor='#FFF9F5'

              )



fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-line')