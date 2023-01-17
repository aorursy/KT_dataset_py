import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Get historical stock

df=pd.read_csv('../input/historical_stock_prices.csv')
# List of column names:

print(list(df.columns.values))
# number of different stocks

print('Number of different stocks: ', len(list(set(df.ticker.unique()))))