import numpy as np

import pandas as pd

from pandas_datareader import data as wb

import matplotlib.pyplot as plt

%matplotlib inline
pf_data = pd.read_csv('../input/GLD_AMZN.csv', index_col = 'Date')
(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10, 5))
log_returns = np.log(pf_data / pf_data.shift(1))

log_returns.mean()*250
log_returns.std()*250**0.5
log_returns.cov()
log_returns.corr()
weights = np.random.random(2)

weights = weights/np.sum(weights)

weights
pfolio_returns = []

pfolio_volatilities = []



for x in range (1000):

    weights = np.random.random(2)

    weights /= np.sum(weights)

    pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250)

    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 250, weights))))

    

pfolio_returns = np.array(pfolio_returns)

pfolio_volatilities = np.array(pfolio_volatilities)



pfolio_returns, pfolio_volatilities
portfolios = pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})

portfolios.head()
portfolios.plot(x='Volatility', y='Return',kind = 'scatter',figsize=(10,6));

plt.xlabel('Expected Volatility')

plt.ylabel('Expected Return')