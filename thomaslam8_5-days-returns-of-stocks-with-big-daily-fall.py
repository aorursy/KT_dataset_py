import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



import time

from datetime import datetime



input_folder = '/kaggle/input/asx200-y18-y20-prices/'

output_folder = '/kaggle/output/'



def get_time(year, month, day):

    dt = datetime(year, month, day)

    return int(round(dt.timestamp()))



def prepare_df(code, columns, time1, time2):

    url = "https://query1.finance.yahoo.com/v7/finance/download/{code}?period1={time1}&period2={time2}&interval=1d&events=history"

    url = url.format(code=code, time1=time1, time2=time2)

    print(code + '...', end='')

    

    df = pd.read_csv(url, parse_dates=True) 

    df = df[columns]

    df.columns = [code + '_' + col for col in columns]

    df.to_csv(output_folder + code + '.csv')

    return df



def load_dfs(df_names, columns, time1, time2):

    dfs = []

    

    for i, df_name in enumerate(df_names):

        dfs.append(prepare_df(df_name, columns, time1, time2))

        

    return dfs



def load_csvs(df_names, columns):

    print('loading...')

    dfs = []

    

    for i, df_name in enumerate(df_names):

        try:

            print(df_name+"...", end="")

            dfs.append(pd.read_csv(input_folder + df_name + '.csv')[[df_name + '_' + col for col in columns]])

        except:

            pass

        

    return dfs



def merge_dfs(dfs, df_names, key_column):

    print('merging...')

    df = dfs[0]

    left_col = df_names[0]+'_'+key_column

    

    for i, df_name in enumerate(df_names):

        if i > 0:

            try:

                print(df_name+"...", end="")

                df = df.merge(dfs[i], how='inner', left_on=left_col, right_on=df_name+'_'+key_column)

            except:

                pass

    

    return df
key_column = 'Date'

columns = [key_column, 'Adj Close']



codes = pd.read_csv(input_folder + 'asx200_codes.csv')['Company'].tolist()

dfs = load_csvs(codes, columns)

df = merge_dfs(dfs, codes, key_column)
df.head()
big_fall_pct = -0.05

days_to_sell = 5

column_names = ["code", "date_buy", "fall_pct", "prev_price", "buy_price", "date_sell", "sell_price", "return"]

returns_df = pd.DataFrame(columns = column_names)



# for each stock

for code in codes:

    col_date = code + '_Date'

    col_close = code + '_Adj Close'

    try: 

        print(code + "...", end="")

        # go through daily closing prices & look for big daily falls

        for i in range(2, len(df.index)-days_to_sell):

            prev_price = df[col_close].iloc[i-1]

            buy_price = df[col_close].iloc[i]

            buy_date = df[col_date].iloc[i]

            fall_pct = buy_price/prev_price - 1.0

            # if big fall

            if fall_pct < big_fall_pct:

                max_sell_price = 0

                max_sell_date = ''

                # assume we could sell stock with max return (within "days to sell" range, i.e. we are not holding the stock forever)

                for j in range(1, days_to_sell+1):

                    sell_price = df[col_close].iloc[i+j]

                    if sell_price > max_sell_price:

                        max_sell_price = sell_price

                        max_sell_date = df[col_date].iloc[i+j]



                new_row = pd.Series([code, buy_date, fall_pct, prev_price, buy_price, max_sell_date, max_sell_price, max_sell_price/buy_price-1], index = returns_df.columns)

                returns_df = returns_df.append(new_row, ignore_index=True)

    except:

        pass



# export the data for those want to do more analysis with Excel, etc.

#returns_df.to_csv(output_folder + 'buy_big_fall_returns.csv')            

            

returns_df
returns_df.describe()
return_stats = returns_df[['code', 'return']].groupby('code').agg(["mean", "median", "min", "max", "std", "count"])

#return_stats.to_csv(output_folder + 'buy_big_fall_returns_summary.csv')

return_stats
return_stats.columns
return_stats[return_stats[("return", "mean")] > 0.03]
from math import floor

import matplotlib.ticker as mtick



# split stocks (_codes) in dataframe (_df) into a number of box plots (_charts)

def show_box_plots(_df, _codes, _charts):

    cnt = len(_codes)



    q = floor(cnt/_charts)



    # print stocks into _charts 

    for i in range(1,_charts+1):

        f, ax = plt.subplots(figsize=(20, 5))

        chart = sns.boxplot(data=_df.loc[:,  _codes[(i-1)*q : i*q]]*100) # multiply by 100 for percent formatter...

        plt.xticks(rotation=65, horizontalalignment='right')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.set(ylim=(-35, 35))

    

    # print the remaining stocks last plot    

    if cnt > _charts*q:

        f, ax = plt.subplots(figsize=(20, 5))

        chart = sns.boxplot(data=_df.loc[:,  _codes[_charts*q : ]]*100) # multiply by 100 for percent formatter...

        plt.xticks(rotation=65, horizontalalignment='right')   

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax.set(ylim=(-35, 35))

plot_data = pd.DataFrame(columns=[])



for code in codes:

    _d = returns_df[returns_df['code'] == code][['return']]

    plot_data = pd.concat([plot_data, _d], ignore_index='True', axis=1)



plot_data.columns = codes
_t = return_stats.T.sort_values(('return', 'median'), axis=1, ascending=False)

sorted_codes_by_medium = _t.columns.tolist()



sorted_codes_by_medium



show_box_plots(plot_data, sorted_codes_by_medium, 3)