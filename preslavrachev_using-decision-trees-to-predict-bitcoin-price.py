import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-03-27.csv')
df.tail()
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df['Weighted_Price_MA60'] = df['Weighted_Price'].rolling(60).mean()
df.tail()
CN_WEIGHTED_PRICE = 'Weighted_Price_MA60'

for days in [1, 7, 14, 30, 60, 90, 120]:
    minutes = days * 24 * 60
    #print(days, df[CN_WEIGHTED_PRICE].rolling(minutes).mean().iloc[-1])
    df['WP_MA' + str(days) + "d"] = df[CN_WEIGHTED_PRICE] / df[CN_WEIGHTED_PRICE].rolling(minutes).max()

df.tail()    
#df.loc[df[CN_NEXT_PCT_CHANGE] >= 0, CN_LABEL] = 1
#df.loc[df[CN_NEXT_PCT_CHANGE] <= 0, CN_LABEL] = -1
future_price = df[CN_WEIGHTED_PRICE].shift(-24)
future_price_pct_gain = (future_price / df[CN_WEIGHTED_PRICE]) - 1

CN_LABEL = 'label'

df.loc[future_price_pct_gain >= 0, CN_LABEL] = 1
df.loc[future_price_pct_gain <= 0, CN_LABEL] = -1
