# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
mi = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/ETFs/spy.us.txt") # Market index

aapl = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/aapl.us.txt") # Apple

googl = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/googl.us.txt") # Alphabet

fb = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/fb.us.txt") # Facebook

amzn = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/amzn.us.txt") # Amazon
aapl['aapl'] = aapl.Close.pct_change()

amzn['amzn'] = amzn.Close.pct_change()

fb['fb'] = fb.Close.pct_change()

googl['googl'] = googl.Close.pct_change()

mi['mi'] = mi.Close.pct_change()
# New DataFrame by using the market index (mi) dates and returns as a starting point

returns=mi[['Date','mi']]#



# Merge returns to returns DataFrame

returns = returns.merge(aapl[['Date','aapl']], left_on='Date', right_on='Date').set_index('Date')

returns = returns.merge(amzn[['Date','amzn']], left_on='Date', right_on='Date').set_index('Date')

returns = returns.merge(fb[['Date','fb']], left_on='Date', right_on='Date').set_index('Date')

returns = returns.merge(googl[['Date','googl']], left_on='Date', right_on='Date').set_index('Date')



# Create the big tech portfolio

returns["bt_portfolio"] = 1/4*(returns.aapl + returns.amzn + returns.fb + returns.googl)



# Clean the DataFrame

returns = returns.dropna()



# Display first five rows.

returns.head()
(returns+1).cumprod().plot()
# daily risk free rate.

rf = (1.02**(1/360))-1 



# New dataframe to stock_data.

stock_data = pd.DataFrame(columns=['security','e_returns','vol', 'sr' ]).set_index('security')



# Calculate volatilities, expected returns and sharpe ratios.

for security in returns.columns:

    vol = returns[security].std() # volatility

    e_r = returns[security].mean() # excpected returns

    sr = (e_r-rf)/vol # Sharpe ratio

    stock_data.loc[security]= [e_r, vol,sr]

    

stock_data.shape
stock_data["sec_type"]=["portfolio","stock","stock","stock","stock","portfolio"]

stock_data.groupby("sec_type").size().plot.pie()
stock_data.groupby("sec_type").vol.mean().sort_values().plot.barh()
stock_data.sr.sort_values().plot.barh()