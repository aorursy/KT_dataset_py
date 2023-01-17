# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
stock_returns = pd.read_csv("../input/small_portfolio.csv")



# Check beginning and end of dataset

print(stock_returns.head())

print(stock_returns.tail())



# Set date as index

stock_returns.set_index('date', inplace=True)

stock_returns
# Calculate mean performance for a four stock portfolio over the period January 2015 through March 2019. 

# The portfolio consists of Proctor & Gamble, Microsoft, JP Morgan and General Electric stocks.



# Calculate percentage returns

returns = stock_returns.pct_change()



# Calculate individual mean returns 

meanDailyReturns = returns.mean()



# Define weights for the portfolio

weights = np.array([0.5, 0.2, 0.2, 0.1])



# Calculate expected portfolio performance

portReturn = np.sum(meanDailyReturns*weights)



# Print the portfolio return

print(portReturn)
# Calculate Portfolio cumulative returns

# The cumulative performance gives you the compounded return at each date in your datase



# Create portfolio returns column

returns['Portfolio'] = returns.dot(weights)



# Calculate cumulative returns

daily_cum_ret=(1+returns).cumprod()



# Plot the portfolio cumulative returns only

fig, ax = plt.subplots()

ax.plot(daily_cum_ret.index, daily_cum_ret.Portfolio, color='purple', label="portfolio")

ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())

plt.legend()

plt.show()
# Measuring risk of a portfolio

# Portfolio variance is a statistical value that assesses the degree of dispersion of the returns of a portfolio.



# The calculation of portfolio variance considers not only the riskiness of individual assets but also the correlation between each pair of assets in the portfolio. 

# Thus, the statistical variance analyzes how assets within a portfolio tend to move together. 

# The general rule of portfolio diversification is the selection of assets with a low or negative correlation between each other.



# Get percentage daily returns

daily_returns = stock_returns.pct_change()



# Assign portfolio weights

weights = np.array([0.05, 0.4, 0.3, 0.25])



# Calculate the covariance matrix 

cov_matrix = (daily_returns.cov())*250



# Calculate the portfolio variance

port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))



# Print the result

print(str(np.round(port_variance, 4) * 100) + '%')