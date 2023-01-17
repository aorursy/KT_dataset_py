# import packages,read csv and combine data from multiple sheets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

data = pd.read_excel('../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx',sheet_name=[0,1])
data = pd.concat([data[0],data[1]],axis=0)
data
data.shape
# Delete canceled orders that start with 'C'
data['Success'] = data['Invoice'].apply(lambda x: 'C' not in str(x))
data = data[data['Success']==True]
data = data.drop('Success',axis=1)
# Delete replenishing orders 
data = data[data['Quantity'] > 0]
# Reformat the InvoiceDate to yyyy-mm-dd
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'].dt.strftime('%Y-%m-%d'))

data.head(1)
# Delete orders on debt (irrelavant on sales)
data = data[data['Price'] > 0]
# Obtain daily sales 
data['TotalPrice'] = data['Quantity']*data['Price']

grp_date = data[['Quantity','InvoiceDate','Price','TotalPrice']].groupby('InvoiceDate')
grp_date = grp_date.sum()

sale = grp_date[['TotalPrice']]
sale.to_csv('DailySalesTrending.csv')
sale
# read daily sales' records
data = pd.read_csv('DailySalesTrending.csv')

data['InvoiceDate'] = data['InvoiceDate'].astype('datetime64[ns]')
data.info()
# set index to be invoce date and find missing dates 
data.set_index(data['InvoiceDate'],drop=False,inplace=True)

missing_date = pd.date_range(start ='2009-12-01', end ='2011-12-09').difference(data.index)
# print dates with missting values
for date in missing_date:
    print(date.year,date.month,date.day,date.dayofweek)
# add missing dates and fill them 
data = data.reindex(pd.date_range(start ='2009-12-01', end ='2011-12-09'))
data.fillna(0,inplace=True)
data['InvoiceDate'] = data.index
# add timestamp 
data['year'] = data['InvoiceDate'].dt.year
data['month'] = data['InvoiceDate'].dt.month
data['day'] = data['InvoiceDate'].dt.day
data['week'] = data['InvoiceDate'].dt.week
data['weekday'] = data['InvoiceDate'].dt.weekday
data['dayofyear'] = data['InvoiceDate'].dt.dayofyear
# Sales trending 
f = plt.figure(figsize=(20,6))
sb.lineplot(x=data.index,y='TotalPrice',data=data).set_title('Sales Trending')
# seasonal decomposition    Ref: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['TotalPrice'], model='additive')
result.plot()
plt.show()
# yearly trending 
sb.boxplot(x='year',y='TotalPrice',data=data)
# outliers within years 
year = data.groupby('year')
for year, df in year:
    IQR = df['TotalPrice'].quantile(0.75) - df['TotalPrice'].quantile(0.25)
    median = df['TotalPrice'].median()
    large_outliers = df[(df['TotalPrice'] > median + 1.5*IQR)]
    print(large_outliers)
# monthly trending 
month = data.groupby('month')
month_sum = month.sum()

f = plt.figure(figsize = (12,4))
ax = sb.lineplot(x=month_sum.index,y='TotalPrice',data=month_sum)
ax.set_title('Sales\' Monthly Trending')

f,axes = plt.subplots(1,2,figsize = (12,5))
sb.violinplot(x='month',y='TotalPrice',data=data,ax=axes[0])
sb.boxplot(x='month',y='TotalPrice',data=data,ax=axes[1])
# sales and day in month 
f,axes = plt.subplots(1,2,figsize = (20,5))
sb.violinplot(x='day',y='TotalPrice',data=data,ax=axes[0])
sb.boxplot(x='day',y='TotalPrice',data=data,ax=axes[1])
# generate heatmap of sales 
data['NormalizedPrice'] = (data['TotalPrice'] - data['TotalPrice'].mean())/data['TotalPrice'].std()

f, axes = plt.subplots(1,3,figsize=(10*3,5))
for i, (year, group) in enumerate(data.groupby('year')):
    hd = group.pivot_table('NormalizedPrice','weekday','week')
    sb.heatmap(hd,ax=axes[i])

data = data.drop('NormalizedPrice',axis=1)
# display auto-correlation graph 
from pandas.plotting import autocorrelation_plot as auto_p
plt.figure(figsize=(20,5))
f = auto_p(data['TotalPrice'])
'''
Check if the time series is stationary by Dickey-Fuller test. 
ref: https://machinelearningmastery.com/time-series-data-stationary-python/
'''
from statsmodels.tsa.stattools import adfuller

X = data['TotalPrice'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
data.to_csv('DataSet.csv')
data