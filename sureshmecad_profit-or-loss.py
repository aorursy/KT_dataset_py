import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/HDFCBANK.csv")
df.head()
df.tail()
df.shape
price = df[['Date','Close']]
price.head(10)
price.info()
price.Date = pd.to_datetime(price.Date, format="%Y-%m-%d")
price = price[(price['Date'] > pd.Timestamp(2019,1,1))]
price.head()
price.tail()
plt.plot(price.Date, price.Close)
plt.figure(figsize=(14,12))
plt.gcf().autofmt_xdate() 
plt.show()
#price.Close = price.Close.mask( price.Date >='2011-07-14', price.Close*5)
price.Close = price.Close.mask( price.Date >='2019-09-19', price.Close*2)

plt.plot(price.Date, price.Close)
plt.figure(figsize=(60,120))
#price.sort_values(by='Date', ascending=False, inplace=True)
price.head()
price['lowest_cumulative_price']=price.Close.cummin()
price['highest_profit']=price.Close-price['lowest_cumulative_price']
price.highest_profit.max()
price['highest_cumulative_price']=price.Close.cummax()
price['highest_loss']=price.Close-price['highest_cumulative_price']
price.highest_loss.min()