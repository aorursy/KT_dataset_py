import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import packages

import pandas as pd

from pandas import datetime

import numpy as np





#to plot within notebook

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

# Above is a special style template for matplotlib, highly useful for visualizing time series data



import seaborn as sns





#setting figure size

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10,10
path = '../input/national-stock-exchange-time-series/'



TCS = pd.read_csv(path + 'tcs_stock.csv', parse_dates=['Date'])



INFY = pd.read_csv(path + 'infy_stock.csv', parse_dates=['Date'])



NIFTY = pd.read_csv(path + 'nifty_it_index.csv', parse_dates=['Date'])





stocks = [TCS, INFY, NIFTY]





TCS.name = 'TCS'

INFY.name = 'INFY'

NIFTY.name = 'NIFTY_IT'
TCS["Date"] = pd.to_datetime(TCS["Date"])

INFY["Date"] = pd.to_datetime(INFY["Date"])

NIFTY["Date"] = pd.to_datetime(NIFTY["Date"])
TCS.head(5)
TCS.shape
INFY.head(5)
NIFTY.head(5)
# Features Generation





def features_build(df):

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year

    df['Month'] = df.Date.dt.month

    df['Day'] = df.Date.dt.day

    df['WeekOfYear'] = df.Date.dt.weekofyear

    

    

    

for i in range(len(stocks)):

    # print(stocks[i])

    features_build(stocks[i])

    
# check for newly added features. 

TCS.shape
TCS.head(3)
# Lets define a function for moving average with rolling window



# Sklearn has seperate function to calculate this: [ DataFrame.rolling(window).mean() ].



def moving_average(series, n):

    """

        Calculate average of last n observations

        

        n - rolling window

    """

    return np.average(series[-n:])





'''

# We can also imlement this user-defined function. But, first we need to isolate both 'Date' and 'Close' columns

under consideration. After that, we need to resample according to week , using:



df.resample('W')



function. Then We will pass the 'Close' column as the 'series' argument to the custom built function.

And, n = rolling window size.





But: We are not doing that here to maintain simplicity in code.



'''
weeks = [4, 16, 28, 40, 52]
def indexing(stock):

    stock.index = stock['Date']

    return stock
indexing(TCS)

indexing(INFY)

indexing(NIFTY)
def plot_time_series(stock, weeks = [4, 16, 28, 40, 52]):

    

    dummy = pd.DataFrame()

    # First Resampling into Weeks format to calculate for weeks

    dummy['Close'] = stock['Close'].resample('W').mean() 

     

    for i in range(len(weeks)):

        m_a = dummy['Close'].rolling(weeks[i]).mean() # M.A using inbuilt function

        dummy[" Mov.AVG for " + str(weeks[i])+ " Weeks"] = m_a

        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(weeks[i], dummy['Close']))

    dummy.plot(title="Moving Averages for {} \n\n" .format(stock.name))

    
plot_time_series(TCS)
plot_time_series(INFY)
plot_time_series(NIFTY)
TCS = TCS.asfreq('D', method ='pad')        # pad-ffill : forward-fill

INFY = INFY.asfreq('D', method ='pad')

NIFTY = NIFTY.asfreq('D', method ='pad')





TCS.name = 'TCS'

INFY.name = 'INFY'

NIFTY.name = 'NIFTY_IT'
def plot_roll_win(stock, win = [10, 75]):

    

    dummy = pd.DataFrame()

    

    dummy['Close'] = stock['Close']

     

    for i in range(len(win)):

        m_a = dummy['Close'].rolling(win[i]).mean() # M.A using predefined function

        dummy[" Mov.AVG for " + str(win[i])+ " Roll Window"] = m_a

        print('Calculated Moving Averages: for {0} weeks: \n\n {1}' .format(win[i], dummy['Close']))

    dummy.plot(title="Moving Averages for {} \n\n" .format(stock.name))
plot_roll_win(TCS)
plot_roll_win(INFY)
plot_roll_win(NIFTY)
def volume_shocks(stock):

    """

    'Volume' - Vol_t

    'Volume next day - vol_t+1

    

    """

    stock["vol_t+1"] = stock.Volume.shift(1)  #next rows value

    

    stock["volume_shock"] = ((abs(stock["vol_t+1"] - stock["Volume"])/stock["Volume"]*100)  > 10).astype(int)

    

    return stock
volume_shocks(TCS)

volume_shocks(INFY)

volume_shocks(NIFTY)
def direction_fun(stock):

    

    # considerng only shock - 1 valued rows.

    # 0 - negative and 1- positive

    if stock["volume_shock"] == 0:

        pass

    else:

        if (stock["vol_t+1"] - stock["Volume"]) < 0:

            return 0

        else:

            return 1
def vol_shock_direction(stock):

    stock['VOL_SHOCK_DIR'] = 'Nan'

    stock['VOL_SHOCK_DIR'] = stock.apply(direction_fun, axis=1)

    return stock
vol_shock_direction(TCS)

vol_shock_direction(INFY)

vol_shock_direction(NIFTY)
def price_shocks(stock):

    """

    'ClosePrice' - Close_t

    'Close Price next day - vol_t+1

    

    """

    stock["price_t+1"] = stock.Close.shift(1)  #next rows value

    

    stock["price_shock"] = (abs((stock["price_t+1"] - stock["Close"])/stock["Close"]*100)  > 2).astype(int)

    

    stock["price_black_swan"] = stock['price_shock'] # Since both had same data anad info/

    

    return stock
price_shocks(TCS)

price_shocks(INFY)

price_shocks(NIFTY)
def direction_fun_price(stock):

    

    # considerng only shock - 1 valued rows.

    # 0 - negative and 1- positive

    if stock["price_shock"] == 0:

        pass

    else:

        if (stock["price_t+1"] - stock["Close"]) < 0:

            return 0

        else:

            return 1
def price_shock_direction(stock):

    stock['PRICE_SHOCK_DIR'] = 'Nan'

    stock['PRICE_SHOCK_DIR'] = stock.apply(direction_fun_price, axis=1)

    return stock
vol_shock_direction(TCS)

vol_shock_direction(INFY)

vol_shock_direction(NIFTY)
def price_shock_wo_vol_shock(stock):

    

    stock["not_vol_shock"]  = (~(stock["volume_shock"].astype(bool))).astype(int)

    stock["price_shock_w/0_vol_shock"] = stock["not_vol_shock"] & stock["price_shock"]

    

    return stock
price_shock_wo_vol_shock(TCS)

price_shock_wo_vol_shock(INFY)

price_shock_wo_vol_shock(NIFTY)
import bokeh

from bokeh.plotting import figure, output_file, show

from bokeh.io import show, output_notebook

from bokeh.palettes import Blues9

from bokeh.palettes import RdBu3

from bokeh.models import ColumnDataSource, CategoricalColorMapper, ContinuousColorMapper

from bokeh.palettes import Spectral11
output_notebook()
def bokeh_plot(stock):

    data = dict(stock=stock['Close'], Date=stock.index)

    

    p = figure(plot_width=800, plot_height=250,  title = 'time series for {}' .format(stock.name), x_axis_type="datetime")

    p.line(stock.index, stock['Close'], color='blue', alpha=0.5)

    

    #show price shock w/o vol shock

    

    p.circle(stock.index, stock.Close*stock["price_shock_w/0_vol_shock"], size=4, legend_label='price shock without vol shock')

    show(p)
output_file("timeseries.html")



bokeh_plot(TCS)

bokeh_plot(INFY)

bokeh_plot(NIFTY)
from statsmodels.tsa.stattools import acf, pacf



def draw_pacf(stock):

    

    lags = 50



    x = list(range(lags))



    p = figure(plot_height=500, title="Partial Autocorrelation PLot {}" .format(stock.name))



    partial_autocorr = pacf(stock["Close"], nlags=lags)

    p.vbar(x=x, top=partial_autocorr, width=0.9)

    show(p)
output_file("PACF.html")



draw_pacf(TCS)

draw_pacf(INFY)

draw_pacf(NIFTY)