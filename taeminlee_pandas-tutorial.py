import pandas as pd

stock = pd.read_csv('/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
stock.head()
stock.tail()
stock = stock.set_index('Date')

stock.tail()
stock['Open']
stock.index
stock.columns.tolist()
stock.shape
stock.shape[0]
stock.shape[1]
stock['Open']
stock[['Open','Close']]
stock.loc['2010-06-29']
stock.iloc[0]
stock.loc[['2010-06-29','2010-06-30']]
stock.iloc[0:2]
stock.sum().head()
stock.mean().head()
stock.median().head()
stock.std().head()
stock.sort_values('Open',ascending = True)
stock.sort_index(ascending = True)
stock['Years'] = pd.DatetimeIndex(stock.index).year

stock['Months'] = pd.DatetimeIndex(stock.index).month

stock.head()
stock.groupby('Years').mean()
stock.groupby(['Years','Months']).mean()
stock.groupby('Years').size()
stock.Years.value_counts()
stock.plot.scatter(x = 'Years', y = 'Close')
# 1. Group the data in the 'Years' category

# 2. Select the category you want to work with

# 3. Summarize (find the mean) the data

# 4. Plot the data in a pie chart



stock.groupby('Years').Close.mean().plot.pie()