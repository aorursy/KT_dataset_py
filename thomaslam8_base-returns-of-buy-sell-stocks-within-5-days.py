import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



import time

from datetime import datetime



input_folder = '/kaggle/input/asx200-y18-y20-prices/'



def get_time(year, month, day):

    dt = datetime(year, month, day)

    return int(round(dt.timestamp()))



def load_df(code, columns):

    print(code + '...', end='')

    

    df = pd.read_csv(input_folder + code + '.csv', parse_dates=True) 

    df = df[columns]

    df.columns = [col for col in columns]



    return df
columns = ['Date', 'Adj Close']

stocks_prices = {}



codes = pd.read_csv(input_folder + 'asx200_codes.csv')['Company'].tolist()

for code in codes:

    try: 

        stocks_prices[code] = load_df(code, [code+'_Date', code+'_Adj Close'])

    except:

        pass
stocks_prices
from math import floor



days_to_sell = 5

days_check_prev_price = 10

trade_return_pct = 0.03

trade_amount = 10000

brokerage = 9.5

stocks_returns = {}



# for each stock

for code, df in stocks_prices.items():

    try:

        print(code + '...', end='')



        column_names = ["code", "date_buy", "buy_price", "num_shares", "date_sell", "sell_price", code + "_return"]

        returns_df = pd.DataFrame(columns = column_names)



        col_date = code + '_Date'

        col_close = code + '_Adj Close'



        # make a trade on each trade day

        for i in range(days_check_prev_price, len(df.index)-days_to_sell):

            prev_price = df[col_close].iloc[i-days_check_prev_price]

            buy_price = df[col_close].iloc[i]

            buy_date = df[col_date].iloc[i]

            num_shares = floor((trade_amount-brokerage)/buy_price)



            # only buy if rising 

            if buy_price >= prev_price:



                # for each trade, can we make return at trade_return_pct within days_to_sell?

                is_sold = False

                for j in range(1, days_to_sell+1):

                    sell_date = df[col_date].iloc[i+j]

                    sell_price = df[col_close].iloc[i+j]

                    roi = (sell_price*num_shares-2*brokerage)/(buy_price*num_shares) - 1

                    if roi > trade_return_pct:

                        new_row = pd.Series([code, buy_date, buy_price, num_shares, sell_date, sell_price, roi], index = returns_df.columns)

                        is_sold = True

                        break



                if not is_sold:

                    new_row = pd.Series([code, buy_date, buy_price, num_shares, sell_date, sell_price, sell_price/buy_price-1], index = returns_df.columns)



                returns_df = returns_df.append(new_row, ignore_index=True)



        stocks_returns[code] = returns_df

    except:

        pass

stocks_returns_df = None



# for each stock

for code, df in stocks_returns.items():

    try:

        print('...' + code + '...', end='')

        print(df[code + "_return"].count(), end='')



        # merge the stocks returns into 

        if stocks_returns_df is None:

            stocks_returns_df = df[['date_buy', code + "_return"]]

        else:

            stocks_returns_df = df[['date_buy', code + "_return"]].merge(stocks_returns_df, how='outer', left_on='date_buy', right_on='date_buy')

    except:

        pass

            

stocks_returns_df.describe()
stocks_returns_df.head()
stocks_returns_stats_df = stocks_returns_df.describe()

stocks_returns_stats_df = stocks_returns_stats_df.sort_values(by =['mean'], axis=1, ascending=False)

stocks_returns_stats_df
from math import floor

import matplotlib.ticker as mtick



# split stocks (_codes) in dataframe (_df) into a number of box plots (_charts)

def show_box_plots(_df, _codes, _charts):

    cnt = len(_codes)



    q = floor(cnt/_charts)



    # print stocks into _charts 

    for i in range(1,_charts+1):

        f, ax = plt.subplots(figsize=(20, 10))

        chart = sns.boxplot(data=_df.loc[:,  _codes[(i-1)*q : i*q]]*100, showmeans=True) # multiply by 100 for percent formatter...

        plt.xticks(rotation=65, horizontalalignment='right')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.set(ylim=(-35, 35))

    

    # print the remaining stocks last plot    

    if cnt > _charts*q:

        f, ax = plt.subplots(figsize=(20, 10))

        chart = sns.boxplot(data=_df.loc[:,  _codes[_charts*q : ]]*100, showmeans=True) # multiply by 100 for percent formatter...

        plt.xticks(rotation=65, horizontalalignment='right')   

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.set(ylim=(-35, 35))
show_box_plots(stocks_returns_df, stocks_returns_stats_df.columns, 15)