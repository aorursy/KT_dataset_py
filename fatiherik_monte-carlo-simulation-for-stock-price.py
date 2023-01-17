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
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
start=dt.datetime(2017,1,3)
end=dt.datetime(2017,11,20)

prices=web.DataReader('AAPL','yahoo', start, end)['Close']
prices
returns=prices.pct_change()
returns
last_price=prices[-1]
last_price
# number of simulation
num_simulations=1000
num_days=252
simulation_df=pd.DataFrame()
for x in range(num_simulations):
    count=0
    daily_vol=returns.std()
    
    price_series=[]
    price=last_price*(1+np.random.normal(0,daily_vol))
    price_series.append(price)
    
    for y in range(num_days):
        if count==251:
            break
        price=price_series[count]*(1+np.random.normal(0,daily_vol))
        price_series.append(price)
        count+=1
    
    simulation_df[x]=price_series
simulation_df
simulation_last_day=simulation_df.iloc[251,:]
simulation_last_day
simulation_last_day.mean()
simulation_last_day.describe()
plt.hist(simulation_last_day, bins=10, color='b')
plt.axvline(x=simulation_last_day.mean(),linewidth=2)
plt.show()
fig=plt.figure(figsize=(19,8))
plt.plot(simulation_df, linewidth=2)
plt.title('Monte Carlo Simulation AAPL')
plt.xlabel('Day')
plt.ylabel('Price')
plt.axhline(y=last_price, color='r', linestyle='-')
plt.show()
