import pandas as pd
netflix = pd.read_csv('../input/NetflixStocks.csv', parse_dates=['date'], index_col='date')

netflix.head(5)
type(netflix.index[0]) #Use this to check the type without assiging the index, then use type(netflix.date[0])
netflix['2019-06'] #As you will see it was a simple work of giving a partial string to get the data when you have date as the index
netflix['2019-06'].close.mean() #This will give you the average stock price for the month of June
netflix.close.resample('M').mean() 
%matplotlib inline

netflix.close.resample('M').mean().plot() #Plotting a graph will help us to understand the basic details.
netflix['2019-06'].close.resample('D').mean().plot()
netflix = netflix.sort_index().asfreq(freq='D', method='pad')
netflix['2019-06'].close.resample('D').mean().plot()
dates = ['2019-10-07', 'Jul 7, 2019', '01/01/2019', '2019.01.01', '2019/07/10', '20170710']

pd.to_datetime(dates)