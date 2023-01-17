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
from pandas_datareader import data as web
from datetime import datetime
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

#create a FAANG portfolio
assets=['FB','AMZN','AAPL','NFLX','GOOG']
# equal weights
weights=np.array([0.2,0.2,0.2,0.2,0.2])
stockStartDate='2013-01-01'
today='2020-03-15'
#today=datetime.today().strftime('%Y=%m-%d')

df=pd.DataFrame()
for stock in assets:
    df[stock]=web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']
df
# Visualize the stocks
my_stocks=df

plt.figure(figsize=(18,9))
for i in my_stocks.columns.values:
    plt.plot(my_stocks[i], label=i, linewidth=2)

plt.title('Portfolio Adj Close Price History')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj Close Price ($)', fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show() 
# daily returns#
returns=df.pct_change()
returns
# annualized covariance matrix
cov_matrix_annual=returns.cov()*252
cov_matrix_annual
# diagonal numbers are variance, other numbers are covariance 
# portfolio variance
port_variance=np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance
#portfolio volatility(=standart deviation)
port_volatility=np.sqrt(port_variance)
port_volatility
# annual portfolio return
portfolioSimpleAnnualReturn=np.sum(returns.mean()*weights)*252
portfolioSimpleAnnualReturn
#expected annual return, volatility(risk), variance

percent_var=str(round(port_variance,2)*100)+'%'
percent_vol=str(round(port_volatility,2)*100)+'%'
percent_ret=str(round(portfolioSimpleAnnualReturn,2)*100)+'%'

print('Expected Annual Return :'+ percent_ret )
print('Annual Volatility/risk :'+ percent_vol )
print('Annual Variance :'+ percent_var )

! pip install PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
# Portfolio optimization

# Calculate the expexted returns and annualised sample covariance matrix of the asset returns
mu=expected_returns.mean_historical_return(df)
S=risk_models.sample_cov(df)

# Optimize the max sharpe ratio
ef=EfficientFrontier(mu,S)
weights=ef.max_sharp()
clean_weights=ef.clean_weights
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
# get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices=get_latest_prices(df)
weights=cleaned_weights
da=DiscreteAllocation(weights, latest_prices, total_portfolio_value=15000)

allocation, leftover=da.lp_portfolio()
print('Discrete Allocation: '+allocation)
print('Funds Remaining: ${:.2f}'.format(leftover))




