# data imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# plot imports
import matplotlib.pyplot as plt
%matplotlib inline

# get finance packages
from matplotlib.finance import candlestick2_ohlc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/EURUSD_15m_BID_01.01.2010-31.12.2016.csv')
TIME = 'Time'
OPEN = 'Open'
CLOSE = 'Close'
HIGH = 'High'
LOW = 'Low'
VOLUME = 'Volume'

ROLLING_AVERAGE = 'Rolling Average'
ROLLING_STD = 'Rolling St Dev'
BOLLINGER_TOP = 'Bollinger Top'
BOLLINGER_BOTTOM = 'Bollinger Bottom'

CROSSED_BOLLINGER_BOTTOM_DOWN = 'Crossed Bollinger Bottom Down'
CROSSED_BOLLINGER_BOTTOM_UP = 'Crossed Bollinger Bottom Up'
CROSSED_BOLLINGER_TOP_DOWN = 'Crossed Bollinger Top Down'
CROSSED_BOLLINGER_TOP_UP = 'Crossed Bollinger Top Up'
class Bollingers:
    
    def __init__(self, df, k, window):
        self.df = df
        self.k = k
        self.window = window
        
        # compute rolling calculations
        self.df[ROLLING_AVERAGE] = self.df[CLOSE].rolling(window=self.window,center=False).mean()
        self.df[ROLLING_STD] = self.df[CLOSE].rolling(window=self.window,center=False).std() 

        # compute bollingers     
        self.df[BOLLINGER_TOP] = self.df.apply(lambda row: self.bollinger('top', row[ROLLING_AVERAGE], row[ROLLING_STD]), axis=1)
        self.df[BOLLINGER_BOTTOM] = self.df.apply(lambda row: self.bollinger('bottom', row[ROLLING_AVERAGE], row[ROLLING_STD]), axis=1)

        # add bools for price crossing bollingers
        self.df[CROSSED_BOLLINGER_BOTTOM_DOWN] = self.df.apply(lambda row: self.crossed_down(row[HIGH], row[CLOSE], row[BOLLINGER_BOTTOM]), axis=1)
        self.df[CROSSED_BOLLINGER_BOTTOM_UP] = self.df.apply(lambda row: self.crossed_up(row[LOW], row[CLOSE], row[BOLLINGER_BOTTOM]), axis=1)
        self.df[CROSSED_BOLLINGER_TOP_DOWN] = self.df.apply(lambda row: self.crossed_down(row[HIGH], row[CLOSE], row[BOLLINGER_TOP]), axis=1)
        self.df[CROSSED_BOLLINGER_TOP_UP] = self.df.apply(lambda row: self.crossed_up(row[LOW], row[CLOSE], row[BOLLINGER_TOP]), axis=1)
        
    def bollinger(self, top_or_bottom, rolling_av, rolling_std):
        if top_or_bottom == 'top':
            return rolling_av + self.k*rolling_std
        elif top_or_bottom == 'bottom':
            return rolling_av - self.k*rolling_std
        else:
            raise ValueError('Expect "top" or "bottom" for top_or_bottom')
    
    def crossed_down(self, High, Close, Bollinger):
        if High >= Bollinger and Close < Bollinger:
            return 1
        else:
            return 0

    def crossed_up(self, Low, Close, Bollinger):
        if Low <= Bollinger and Close > Bollinger:
            return 1
        else:
            return 0
        
    def visualise(self, start_row, end_row):
        df_sample = self.df[start_row:end_row]
        time_range = range(0, len(df_sample[TIME]))

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)

        candlestick2_ohlc(ax, 
                          opens=df_sample[OPEN], 
                          closes=df_sample[CLOSE], 
                          highs=df_sample[HIGH],
                          lows=df_sample[LOW],
                          width=1,
                          colorup='g', 
                          colordown='r',
                          alpha=0.75)

        ax.plot(time_range, df_sample[ROLLING_AVERAGE])
        ax.plot(time_range, df_sample[BOLLINGER_TOP])
        ax.plot(time_range, df_sample[BOLLINGER_BOTTOM])

        plt.ylabel("Price")
        plt.xlabel("Time Periods")
        plt.legend()

        plt.show()
bollingers = Bollingers(df, 3, 20)
bollingers.visualise(300,400)
class Strategy:
    
    def __init__(self):
        self.position = 'none'
        self.balance = 0
        self.bought_at = 0
        self.win_loss_array = []
        self.profit_array = []
        
    def long(self, row):
        if self.position == 'none':
            self.position = 'long'
            self.bought_at = row[CLOSE]
        
    def short(self, row):
        if self.position == 'none':
            self.position = 'short'
            self.bought_at = row[CLOSE]
        
    def close_position(self, row):
        
        if self.position == 'short':
            profit_amount = row[CLOSE] - self.bought_at
        elif self.position == 'long':
            profit_amount = self.bought_at - row[CLOSE]
        else:
            profit_amount = 0
            print ('Tried to close a position when none were open')
            
        is_profit = profit_amount > 0
            
        self.balance += profit_amount - 0.000072 # just an average spread i found online
        self.profit_array.append(profit_amount)
        self.win_loss_array.append(is_profit)
        self.position = 'none'
        self.bought_at = 0
        
    def print_results(self):
        print ('Final balance: ' + str(self.balance) + ' = ' + str(self.balance*1000) + ' pips')
        print ('Win/Loss ratio: ' + str(sum(self.win_loss_array)/len(self.win_loss_array)*100))
        print ('Biggest win: ' + str(np.amax(self.profit_array)))
        print ('Biggest loss: ' + str(np.amin(self.profit_array)))
def strategy_1(bollinger_df):
    
    strategy = Strategy()

    for index, row in bollinger_df.iterrows():
        if row[CROSSED_BOLLINGER_BOTTOM_DOWN] == 1 and strategy.position == 'none':
            strategy.long(row)
            
        if row[CROSSED_BOLLINGER_TOP_UP] == 1 and strategy.position == 'none':
            strategy.short(row)

        if row[CROSSED_BOLLINGER_BOTTOM_UP] == 1 and strategy.position == 'long':
            strategy.close_position(row)

        if row[CROSSED_BOLLINGER_TOP_DOWN] == 1 and strategy.position == 'short':
            strategy.close_position(row)
            
    
    return strategy.print_results()
strategy_1(bollingers.df)
