

import numpy as np 

import pandas as pd 

import datetime

import os



#define a conversion function for the native timestamps in the csv file

def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))



print('Data listing...')

print(os.listdir('../input'))



bitstamp_csv = 'bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv'



print('Using bitstampUSD_1-min_data...')

data = pd.read_csv('../input/' + bitstamp_csv, parse_dates=True, date_parser=dateparse, index_col=[0])
print(data.tail())
# Null entries

print( data.isnull().sum())
# First thing is to fix the data for bars/candles where there are no trades. 

# Volume/trades are a single event so fill na's with zeroes for relevant fields...

data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(value=0, inplace=True)



# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so

# lets fill forwards those values...

data['Open'].fillna(method='ffill', inplace=True)

data['High'].fillna(method='ffill', inplace=True)

data['Low'].fillna(method='ffill', inplace=True)

data['Close'].fillna(method='ffill', inplace=True)



print(data.tail())
# The first thing we need are our trading signals. The Turtle strategy was based on daily data and

# they used to enter breakouts (new higher highs or new lower lows) in the 22-55 day range roughly.



signal_lookback = 55 * 24 * 60 # days * hours * minutes



# here's our signal columns

data['Buy'] = np.zeros(len(data))

data['Sell'] = np.zeros(len(data))



# core strategy: enter when there is a price breakout over the rolling mean, exit in the same fashion

data['RollingMax'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).max()

data['RollingMin'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).min()

data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1

data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1



# lets now take a look and see if its doing something sensible

import matplotlib

import matplotlib.pyplot as plt



fig,ax1 = plt.subplots(1,1)

ax1.plot(data['Close'])

ax1.plot(data['RollingMax'], color = 'g')

ax1.plot(data['RollingMin'], color = 'r')

y = ax1.get_ylim()

x = ax1.get_xlim()

ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])

ax1.set_xlim(x[0]  ,x[1])


# lets now take a look and see if its doing something sensible

import matplotlib

import matplotlib.pyplot as plt



fig,ax1 = plt.subplots(1,1)

ax1.plot(data['Close'])

y = ax1.get_ylim()

ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])



ax2 = ax1.twinx()

ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))

ax2.plot(data['Buy'], color='#77dd77')

ax2.plot(data['Sell'], color='#dd4444')
number_of_buys = data['Buy'].sum()

number_of_sell =- data['Sell'].sum()

print("number_of_buy: ", number_of_buys)

print("number_of_sell: ", number_of_sell)
buy_prices = data[data['Buy'] == 1]['Close']

sell_prices = data[data['Sell'] == -1]['Close']
buy_prices = pd.Series.to_frame(buy_prices)

buy_prices = buy_prices.rename(columns={'Close': 'Close_buy'})

print(buy_prices.shape)

sell_prices = pd.Series.to_frame(sell_prices)

sell_prices = sell_prices.rename(columns={'Close': 'Close_sell'})

print(sell_prices.shape)

total_size = sell_prices.shape[0] + buy_prices.shape[0]

print("total size: ", total_size)
buy_prices['buy_signal'] = 1

sell_prices['sell_signal'] = -1
# dataframe with dates as indices, buy and sell prices 

#and buy-sell token to understand if we need to buy or sell

buy_and_sell = buy_prices.join(sell_prices, how='outer')

# it's fine to have NaNs in other columns

print(buy_and_sell.head())

print(buy_and_sell.tail())
# compute revenue of turtle trading method

def compute_revenue(initial_holdings,

                    # A possible strategy is to always buy and sell fixed amounts. In the actual turtle 

                    # strategy these were dynamical, adjusted on the volatility of the stocks 

                    # If use_fixed_USD_amount = False then we use an even simpler (but less realistic)

                    # strategy, that is always buy or sell 1 BTC (when there is a signal),

                    #no matter how much it cost.

                    use_fixed_USD_amount = False, fixed_amount_buy = 100, fixed_amount_sell = 100,

                    # Starting from initial_holdings, could we borrow additional USD?

                    allow_negative_usd_holdings = False):

    

    usd_holdings = initial_holdings

    btc_holdings_value = 0

    total_holdings_value = []

    dates = []

    number_of_btc = 0



    for i in range(len(buy_and_sell)):

        

        if( buy_and_sell.iloc[i]['buy_signal'] == 1):

            

            current_price = buy_and_sell.iloc[i]['Close_buy']

            if (( usd_holdings - current_price > 0) 

               or (( usd_holdings - current_price < 0) and (allow_negative_usd_holdings == True))):

                

                if use_fixed_USD_amount:

                    btc_to_buy = float(fixed_amount_buy)/current_price

                    number_of_btc += btc_to_buy

                    usd_holdings = usd_holdings - fixed_amount_buy

                else:

                    number_of_btc += 1

                    usd_holdings = usd_holdings - current_price   

                

            

        if( buy_and_sell.iloc[i]['sell_signal'] == -1):

        

            current_price = buy_and_sell.iloc[i]['Close_sell']

            

            if (use_fixed_USD_amount and (number_of_btc > 0)):

                btc_to_sell = float(fixed_amount_sell)/current_price

                if (btc_to_sell > number_of_btc):

                    number_of_btc += btc_to_sell

                    usd_holdings = usd_holdings + fixed_amount_sell

            elif (number_of_btc > 0):

                number_of_btc -= 1

                usd_holdings = usd_holdings + current_price   

        

        btc_holdings_value = number_of_btc * current_price

        total_holdings_value.append(btc_holdings_value + usd_holdings)

        dates.append(buy_and_sell.index[i])

    

    

    print("Number of BTC remaining: ", number_of_btc)

    print("btc_holdings_value: ", btc_holdings_value)

    print("usd_holdings_value: ",usd_holdings)

    print("total_holdings_value: ",total_holdings_value[-1])

    print("Total Gain (ROI): ", (total_holdings_value[-1]/float(initial_holdings) -1) *100 , "%")

    

    return btc_holdings_value, usd_holdings, total_holdings_value, number_of_btc, dates

    

                



    

    
def plot_revenue(dates, total_holdings_value):

    import matplotlib.dates as mdates

    fig, ax = plt.subplots()

    ax.plot(dates, total_holdings_value)

    plt.ylabel('total_holdings_value')



    years = mdates.YearLocator()   # every year

    ax.xaxis.set_major_locator(years)



    #datemin = datetime.date(2015, 6, 1)

    #datemax = datetime.date(2017, 1, 1)

    #ax.set_xlim(datemin, datemax)



    fig.autofmt_xdate()

    plt.show()
one_milion = 1000000

initial_holdings = 1 * one_milion



# always buy/sell 1 BTC

_, _, total_holdings_value1, _, dates1 = compute_revenue(initial_holdings ,

                                                       allow_negative_usd_holdings = False)

plot_revenue(dates1, total_holdings_value1)
# always buy/sell a fixed amount

initial_holdings = 1 * one_milion



_, _, total_holdings_value2, _, dates2 = compute_revenue(initial_holdings ,

                                                       use_fixed_USD_amount = True, 

                                                       fixed_amount_buy = 100, 

                                                       fixed_amount_sell = 100)

plot_revenue(dates2, total_holdings_value2)