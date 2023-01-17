import pandas as pd
stocks = pd.read_csv('../input/stocks.csv',parse_dates = ['Date'])
stocks.groupby('Symbol').mean()
1# Use the Query Method



stocks.groupby('Symbol').mean().query('Close > 100')
2# Create an intermediate variable



temp = stocks.groupby('Symbol').mean()

temp[temp.Close > 100]