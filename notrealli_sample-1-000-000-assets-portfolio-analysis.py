# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stocks = pd.read_csv('/kaggle/input/capital-asset-pricing-model-capm/stock.csv')

stocks.sort_values(by='Date')

stocks
def plot(df, title):

    fig = px.line(df.iloc[:, 1:].set_index(df['Date']), title=title)

    fig.show()

plot(stocks, 'Prices Over Time')
# generate normailized weights for the 9 assets

np.random.seed()



weights = np.array(np.random.random(9))

weights = weights / weights.sum()

weights
# normalize the asset prices based on the initial price

def normalize(df):

    copy = df.copy()

    copy.iloc[:, 1:] = copy.iloc[:, 1:] / copy.iloc[0, 1:]

    return copy



stocks_n = normalize(stocks)

stocks_n
plot(stocks_n, 'Price Variation Over Time')
stocks_n.info()
portfolio = stocks_n.copy()

portfolio.iloc[:, 1:] = portfolio.iloc[:, 1:] * weights * 1000000

portfolio['Total Value $'] = portfolio.iloc[:, 1:].sum(axis=1)

portfolio
shifted = pd.Series(portfolio['Total Value $'][0]).append(portfolio['Total Value $'][:-1], ignore_index=True)

portfolio['Daily Return %'] = (portfolio['Total Value $'] - shifted) / shifted * 100

portfolio
fig = px.line(x=portfolio['Date'], y=portfolio['Daily Return %'], title='Portfolio Daily Return Variation', labels={'x': 'Date', 'y': 'Daily Change'})

fig.show()
fig = px.line(portfolio.iloc[:, 1:-2].set_index(portfolio['Date']), title='Individual Securities Worth')

fig.show()
original_prices = portfolio.iloc[0, 1:-2]

current_prices = portfolio.iloc[-1, 1:-2]



cum_returns = (current_prices - original_prices) / original_prices * 100

cum_returns.astype(float).round(2)
avg_daily_return = portfolio['Daily Return %'].mean()

avg_daily_return
standard_deviation = portfolio['Daily Return %'].std()

standard_deviation
(avg_daily_return * 252) / (standard_deviation * np.sqrt(252))