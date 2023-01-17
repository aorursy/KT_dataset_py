import os

from pandas_datareader import data, wb

import pandas as pd

import seaborn as sns

import numpy as np

import pandas_datareader.data as web

import datetime

import matplotlib.pyplot as plt

import cufflinks as cf



start = "2019-01-01"

end = "2020-04-03"

wixDF = web.DataReader('WIX', 'yahoo', start=start, end=end)

celcomDF = web.DataReader('CEL', 'yahoo', start=start, end=end)

elbitDF = web.DataReader('ESLT', 'yahoo', start=start, end=end)

tevaDF = web.DataReader('TEVA', 'yahoo', start=start, end=end)

delekDF = web.DataReader('DK', 'yahoo', start=start, end=end)
companiesData = [wixDF, celcomDF, elbitDF, tevaDF, delekDF]

tickers = "WIX CEL ESLT TEVA DK".split()

companies_stocks = pd.concat(companiesData, keys=tickers, axis=1)

companies_stocks.columns.names = ['Company Ticker', 'Stock Info']

companies_stocks.head()
# MAX CLOSE DATA FRAME



amount_ser = pd.DataFrame(companies_stocks.xs(key="Close", axis=1, level='Stock Info').max())

date_ser = pd.DataFrame(companies_stocks.xs(key="Close", axis=1, level='Stock Info').idxmax())



max_closeDF = pd.merge(amount_ser, date_ser, on='Company Ticker')

max_closeDF.columns = ['Close', 'Date']

max_closeDF





# MIN CLOSE DATA FRAME



amount_ser = pd.DataFrame(companies_stocks.xs(key="Close", axis=1, level='Stock Info').min())

date_ser = pd.DataFrame(companies_stocks.xs(key="Close", axis=1, level='Stock Info').idxmin())



min_closeDF = pd.merge(amount_ser, date_ser, on='Company Ticker')

min_closeDF.columns = ['Close', 'Date']

min_closeDF
minmax = pd.merge(min_closeDF, max_closeDF, on='Company Ticker')

minmax.columns = ['Min Close', 'Min Date', 'Max Close', 'Max Date']

minmax
# RETURNS

returns = pd.DataFrame(companies_stocks.xs(key='Close', axis=1, level=1).pct_change())

fig= returns.iplot(asFigure=True,kind='box',title='Returns Box Plot: 2019/01/01-2020/04/03')

fig.show()
for ticker in tickers:

    sns.distplot(returns[ticker], hist=False, label=ticker, axlabel="Return")

plt.title("Return's Distribution: 2019/01/01-2020/04/03")

plt.ylabel('Observations')

   

close = pd.DataFrame(companies_stocks.xs(key='Close', axis=1, level=1), companies_stocks.index, tickers)

fig1 = close.iplot(asFigure=True, xTitle="Date",

                   yTitle="Close Price", title="Close by Date")

fig1.show()


WIX19close = pd.DataFrame()

WIX19close['Close'] = (close['WIX'].loc['2019':])

WIX19close['average-30-day'] = None

for i in range(30, len(WIX19close)):

    WIX19close['average-30-day'].iloc[i] = WIX19close['Close'].iloc[i - 30:i].mean()



fig2 = WIX19close.iplot(asFigure=True, xTitle="Date",

                        yTitle="Close", title="WIX: Close by Date")

fig2.show()


sns.clustermap(close['2019':].corr(), cmap='coolwarm', linewidths=3, linecolor='black', annot=True)

plt.title('Correlation Since 01/01/2019', loc='center')



plt.tight_layout()

plt.show()