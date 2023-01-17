# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df2 = pd.read_csv('../input/us-historical-stock-prices-with-earnings-data/stocks_latest/stock_prices_latest.csv', delimiter=',',

                  parse_dates=['date'], index_col='date')

df2.head()
tech_stocks = df2[df2.symbol.isin(['BABA','AAPL','AMZN','MSFT','JD','QQQ'])][['symbol','close_adjusted']]

tech_pivot = tech_stocks.pivot_table(index='date', columns='symbol', values='close_adjusted')

tech_final = tech_pivot.dropna(how='any',axis=0)
tech_final.info()
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



tech_final.plot(title='Historical Stock Prices',linewidth=1);
stock_returns = tech_final[['BABA','AAPL','AMZN','MSFT','JD']].pct_change()

stock_returns.plot(subplots=True, linewidth=1);
benchmark_returns = tech_final['QQQ'].pct_change()

benchmark_returns.plot(linewidth=1);
excess_returns = stock_returns.sub(benchmark_returns, axis=0)

avg_excess_returns = excess_returns.mean()

plt.subplot(1,2,1) 

avg_excess_returns.plot(kind='bar', title='Mean of the Return Diff');



std_excess_returns = excess_returns.std()

plt.subplot(1,2,2) 

std_excess_returns.plot(kind='bar', title='Std of the Return Diff');

plt.tight_layout()
daily_sharpe_ratio = avg_excess_returns.div(std_excess_returns)



annual_factor = np.sqrt(252)

annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)



annual_sharpe_ratio.sort_values().plot(kind='bar', title='Annualized Sharpe Ratio: Tech Stocks vs QQQ',);
annual_sharpe_ratio.sort_values()