from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
symbol = ['AAPL', 'FB', 'GOOGL']
source = 'yahoo'
start_date = '2018-01-01'
end_date = '2019-04-24'
stock = data.DataReader(symbol, source, start_date, end_date)
stock.head()
# stock.to_pickle('data_cache/stock')
# stock = pd.read_pickle('data_cache/stock')
# stock.head()
stock['Close'].head()
## Your code below


## -- Solution code
closingprice = stock['Close']
quarter1 = pd.date_range(start="2019-01-01", end="2019-03-31")
closingprice = closingprice.reindex(quarter1)
closingprice.head(8)
## Your code below


## -- Solution code
stock.head(8)
stock.stack().head(8)
## Your code below


## -- Solution code
aapl = stock.xs('AAPL', level='Symbols', axis=1)
aapl.head()
aapl.shape
aapl_melted = aapl.melt()
aapl_melted.head()
aapl_melted.shape
aapl.reset_index().melt(id_vars='Date')[325:333]
march = pd.date_range(start="2018-03-01", end="2019-03-31")
aapl = stock.xs('AAPL', level='Symbols', axis=1)
aapl = aapl.reindex(march)
aapl.head(10)
## Your code below


## -- Solution code
aapl.head()
aapl['Volume'].plot()
print(plt.style.available)
plt.style.use('default')
aapl.loc[:, ['High', 'Low', 'Adj Close']].plot()
volume = stock.xs('Volume', level='Attributes', axis=1)
volume = volume.round(2)
volume.head()
volume_melted = volume.melt()
volume_melted.head()
volume_melted.groupby('Symbols').mean()
volume_melted.groupby('Symbols').mean().plot(kind='bar')
aapl = stock.xs('AAPL', level='Symbols', axis=1)
aapl = aapl.round(2)
aapl['Close_Diff'] = aapl['Close'].diff()
aapl['Weekday'] = aapl.index.weekday_name
aapl['Month'] = aapl.index.month_name()
aapl.tail()
aapl.groupby('Weekday').mean()
aapl.groupby('Weekday').mean()['Close_Diff'].plot(kind='bar')
aapl.groupby('Weekday').mean()['Close_Diff'].\
sort_values(ascending=False).\
plot(kind='bar')
wday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

aapl_wday = aapl.groupby('Weekday').mean()['Close_Diff']
aapl_wday.index = pd.CategoricalIndex(aapl_wday.index,\
                                      categories=wday,\
                                      ordered=True)

aapl_wday.sort_index().plot(kind='bar')
closingprice = stock['Close']
quarter1 = pd.date_range(start="2019-01-01", end="2019-03-31")
closingprice = closingprice.reindex(quarter1)
closingprice.head(8)
closingprice.isna().sum()
## Your code below



## -- Solution code

months = ["January", "February", "March"]
closingprice['Month'] = closingprice.index.month_name()
average_closing = closingprice.groupby('Month').mean()
average_closing.index = pd.CategoricalIndex(average_closing.index,\
                                            categories=months,\
                                            ordered=True)

average_closing.sort_index().plot(kind='bar')
stock.stack().reset_index().groupby('Symbols').agg({
    'Close': 'mean',
    'High': 'max',
    'Low': 'min'
})
stock.stack().reset_index().groupby('Symbols').agg({
    'Close': 'median',
    'High': 'max',
    'Low': 'min'
}).plot(kind='bar')
import datetime

stock['YearMonth'] = pd.to_datetime(stock.index.date).to_period('M')
monthly_closing = stock.groupby('YearMonth').mean().loc[:,['Close','Low', 'High']]
monthly_closing.head()
## Your code below


## -- Solution code