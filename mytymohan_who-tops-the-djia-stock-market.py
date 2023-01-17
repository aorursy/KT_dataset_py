import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from pylab import rcParams

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 

# Above is a special style template for matplotlib, highly useful for visualizing time series data

%matplotlib inline

import numpy as np

import datetime as dt

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())

np.set_printoptions(suppress=True)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
all_stock = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv")
top10_query = """SELECT *

                 FROM (SELECT Name, AVG(Volume) as Avg

                       FROM all_stock

                       GROUP BY Name

                       ORDER BY AVG(Volume) DESC )

                 LIMIT 10;"""





top10 = pysqldf(top10_query)
stock_10_query = """SELECT * FROM all_stock

                    where Name in ('AAPL', 'GE', 'MSFT', 'INTC', 'CSCO',

                                   'PFE', 'JPM', 'AABA', 'XOM', 'KO')"""



stock_10 = pysqldf(stock_10_query)
stock_10.describe()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
#Missing value proportion in the dataset

missing_values_table(stock_10)
#Dropping all the rows which has NA values

stock_10.dropna(inplace=True)
stock_10.dtypes
#Converting Object to Date format for Date column

stock_10['Date'] = pd.to_datetime(stock_10['Date'])
stock_10['Year'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%Y'))

stock_10['Mon'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%b'))

stock_10['Mon-Year'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%b-%Y'))
year_trend_query = """SELECT Name, Year, AVG(Volume) as Avg

                      from stock_10

                      GROUP BY Name, Year

                      ORDER BY Name, Year;"""



year_trend = pysqldf(year_trend_query)



AAPL_trend = year_trend[year_trend.Name == 'AAPL']

GE_trend = year_trend[year_trend.Name == 'GE']

MSFT_trend = year_trend[year_trend.Name == 'MSFT']

INTC_trend = year_trend[year_trend.Name == 'INTC']

CSCO_trend = year_trend[year_trend.Name == 'CSCO']

PFE_trend = year_trend[year_trend.Name == 'PFE']

JPM_trend = year_trend[year_trend.Name == 'JPM']

AABA_trend = year_trend[year_trend.Name == 'AABA']

XOM_trend = year_trend[year_trend.Name == 'XOM']

KO_trend = year_trend[year_trend.Name == 'KO']
data = [

    go.Scatter(

        x=AAPL_trend['Year'], 

        y=AAPL_trend['Avg'],

        name='Apple'

    ),

    go.Scatter(

        x=GE_trend['Year'], 

        y=GE_trend['Avg'],

        name='GE'

    ),

        go.Scatter(

        x=MSFT_trend['Year'], 

        y=MSFT_trend['Avg'],

        name='Microsoft'

    ),

    go.Scatter(

        x=INTC_trend['Year'], 

        y=INTC_trend['Avg'],

        name='Intel'

    ),

        go.Scatter(

        x=CSCO_trend['Year'], 

        y=CSCO_trend['Avg'],

        name='Cisco'

    ),

    go.Scatter(

        x=PFE_trend['Year'], 

        y=PFE_trend['Avg'],

        name='Pfizer'

    ),

        go.Scatter(

        x=JPM_trend['Year'], 

        y=JPM_trend['Avg'],

        name='JPMorgan'

    ),

    go.Scatter(

        x=AABA_trend['Year'], 

        y=AABA_trend['Avg'],

        name='Altaba'

    ),

        go.Scatter(

        x=XOM_trend['Year'], 

        y=XOM_trend['Avg'],

        name='Exxon Mobil'

    ),

    go.Scatter(

        x=KO_trend['Year'], 

        y=KO_trend['Avg'],

        name='Coca-Cola'

    )

]



layout = go.Layout(

    xaxis=dict(type='category', title='Year'),

    yaxis=dict(title='Average Volume of Stocks Traded'),

    title="Average Stock Volume Trend - Top 10 Companies stock over 2006 - 2017"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='line-chart')
year_trend_query = """SELECT Name, Year, AVG(Close) as Avg

                      from stock_10

                      GROUP BY Name, Year

                      ORDER BY Name, Year;"""



year_trend = pysqldf(year_trend_query)



AAPL_trend = year_trend[year_trend.Name == 'AAPL']

GE_trend = year_trend[year_trend.Name == 'GE']

MSFT_trend = year_trend[year_trend.Name == 'MSFT']

INTC_trend = year_trend[year_trend.Name == 'INTC']

CSCO_trend = year_trend[year_trend.Name == 'CSCO']

PFE_trend = year_trend[year_trend.Name == 'PFE']

JPM_trend = year_trend[year_trend.Name == 'JPM']

AABA_trend = year_trend[year_trend.Name == 'AABA']

XOM_trend = year_trend[year_trend.Name == 'XOM']

KO_trend = year_trend[year_trend.Name == 'KO']
data = [

    go.Scatter(

        x=AAPL_trend['Year'], 

        y=AAPL_trend['Avg'],

        name='Apple'

    ),

    go.Scatter(

        x=GE_trend['Year'], 

        y=GE_trend['Avg'],

        name='GE'

    ),

        go.Scatter(

        x=MSFT_trend['Year'], 

        y=MSFT_trend['Avg'],

        name='Microsoft'

    ),

    go.Scatter(

        x=INTC_trend['Year'], 

        y=INTC_trend['Avg'],

        name='Intel'

    ),

        go.Scatter(

        x=CSCO_trend['Year'], 

        y=CSCO_trend['Avg'],

        name='Cisco'

    ),

    go.Scatter(

        x=PFE_trend['Year'], 

        y=PFE_trend['Avg'],

        name='Pfizer'

    ),

        go.Scatter(

        x=JPM_trend['Year'], 

        y=JPM_trend['Avg'],

        name='JPMorgan'

    ),

    go.Scatter(

        x=AABA_trend['Year'], 

        y=AABA_trend['Avg'],

        name='Altaba'

    ),

        go.Scatter(

        x=XOM_trend['Year'], 

        y=XOM_trend['Avg'],

        name='Exxon Mobil'

    ),

    go.Scatter(

        x=KO_trend['Year'], 

        y=KO_trend['Avg'],

        name='Coca-Cola'

    )

]



layout = go.Layout(

    xaxis=dict(type='category', title='Year'),

    yaxis=dict(title='Average Close Price of the Stocks'),

    title="Average Stock Price based on Close Price - Top 10 Companies stock over 2006 - 2017"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='line-chart')
stk = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", index_col='Date', 

                  parse_dates=['Date'])



app_stk = stk.query('Name == "AAPL"')
app_stk['2006':'2017'].plot(subplots=True, figsize=(10,12))

plt.title('Apple stock trend from 2006 to 2017')

plt.savefig('app_stk.png')

plt.show()

app_stk['Change'] = app_stk.Close.div(app_stk.Close.shift())

app_stk['Change'].plot(figsize=(20,8))
app_stk['Return'] = app_stk.Change.sub(1).mul(100)

app_stk['Return'].plot(figsize=(20,8))
app_stk.Close.diff().plot(figsize=(20,6))
stk = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", index_col='Date', 

                  parse_dates=['Date'])

ms_stk = stk.query('Name == "MSFT"')

itl_stk = stk.query('Name == "INTC"')
# Plotting before normalization

app_stk.Close.plot()

ms_stk.Close.plot()

itl_stk.Close.plot()

plt.legend(['Apple','Microsoft', 'Intel'])

plt.show()
# Normalizing and comparison

# Both stocks start from 100

norm_app_stk = app_stk.Close.div(app_stk.Close.iloc[0]).mul(100)

norm_ms_stk_stk = ms_stk.Close.div(ms_stk.Close.iloc[0]).mul(100)

norm_itl_stk_stk = itl_stk.Close.div(itl_stk.Close.iloc[0]).mul(100)

norm_app_stk.plot()

norm_ms_stk_stk.plot()

norm_itl_stk_stk.plot()

plt.legend(['Apple','Microsoft', 'Intel'])

plt.show()
# Rolling window functions

rolling_app = app_stk.Close.rolling('90D').mean()

app_stk.Close.plot()

rolling_app.plot()

plt.legend(['Close','Rolling Mean'])

# Plotting a rolling mean of 90 day window with original Close attribute of Apple stocks

plt.show()
# Expanding window functions

app_stk_mean = app_stk.Close.expanding().mean()

app_stk_std = app_stk.Close.expanding().std()

app_stk.Close.plot()

app_stk_mean.plot()

app_stk_std.plot()

plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])

plt.show()
# OHLC chart of Apple for December 2016

trace = go.Ohlc(x=app_stk['12-2016'].index,

                open=app_stk['12-2016'].Open,

                high=app_stk['12-2016'].High,

                low=app_stk['12-2016'].Low,

                close=app_stk['12-2016'].Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# OHLC chart of Apple stock for 2016

trace = go.Ohlc(x=app_stk['2016'].index,

                open=app_stk['2016'].Open,

                high=app_stk['2016'].High,

                low=app_stk['2016'].Low,

                close=app_stk['2016'].Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# OHLC chart of Apple stock 2006 - 2017

trace = go.Ohlc(x=app_stk.index,

                open=app_stk.Open,

                high=app_stk.High,

                low=app_stk.Low,

                close=app_stk.Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# Candlestick chart of Apple for December 2016

trace = go.Candlestick(x=app_stk['12-2016'].index,

                open=app_stk['12-2016'].Open,

                high=app_stk['12-2016'].High,

                low=app_stk['12-2016'].Low,

                close=app_stk['12-2016'].Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Candlestick chart of Apple for 2016

trace = go.Candlestick(x=app_stk['2016'].index,

                       open=app_stk['2016'].Open,

                       high=app_stk['2016'].High,

                       low=app_stk['2016'].Low,

                       close=app_stk['2016'].Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Candlestick chart of Apple stock 2006 - 2017

trace = go.Candlestick(x=app_stk.index,

                       open=app_stk.Open,

                       high=app_stk.High,

                       low=app_stk.Low,

                       close=app_stk.Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Consider the Apple stock w.r.t Close price

app_stk["Close"].plot(figsize=(16,8))
# Decomposition of Apple Stock based on Close price

rcParams['figure.figsize'] = 11, 9

decomposed_app_stk = sm.tsa.seasonal_decompose(app_stk["Close"],freq=360) # The frequncy is annual

figure = decomposed_app_stk.plot()

plt.show()
# Plotting white noise

rcParams['figure.figsize'] = 16, 6

white_noise = np.random.normal(loc=0, scale=1, size=1000)

# loc is mean, scale is variance

plt.plot(white_noise)
# Plotting autocorrelation of white noise

plot_acf(white_noise,lags=20)

plt.show()
# The original non-stationary plot

rcParams['figure.figsize'] = 16, 6

decomposed_app_stk.trend.plot()
# The new stationary plot

decomposed_app_stk.trend.diff().plot()