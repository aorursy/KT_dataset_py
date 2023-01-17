import pandas as pd
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sqlite3
import warnings
warnings.filterwarnings('ignore')
data = pd.read_excel('../input/XLS_salesorder.xlsx')
data.head()
data.shape
data[data.duplicated(keep = False) == True]
data = data.drop_duplicates()
data.describe()
data['monthordered'] = pd.to_datetime(list(data['dateordered']), format = "%Y/%m/%d").strftime('%b') 
data_return = data.groupby(['monthordered', 'orderstatus']).sum().reset_index()

not_returned = data_return.loc[data_return['orderstatus'] == 'complete', :]
returned = data_return.loc[data_return['orderstatus'] == 'returned', :]

data_new = pd.merge(not_returned, returned, on = 'monthordered', suffixes = ('_complete', '_returned'))
del [not_returned, returned]
data_new['orders_total'] = data_new['orders_returned'] + data_new['orders_complete']
data_new['return_rate'] = data_new['orders_returned'] / data_new['orders_total']

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
data_new['monthordered'] = pd.Categorical(data_new['monthordered'], categories=months, ordered=True)
data_new = data_new.sort_values(by = 'monthordered')
data_new
x= data_new['monthordered']
y1=data_new['orders_total']
y2=data_new['orders_returned'] 

idx = np.arange(len(x))
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(idx, y1, 'orange', label = 'order_total')
ax2.plot(idx, y2, 'teal', label = 'order_returned')
 
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')

plt.xticks((0,1,2,3,4),('Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
ax1.set_xlabel('month ordered')
ax1.set_ylabel('orders_total', color='orange')
ax2.set_ylabel('orders_returned', color='teal')
plt.title('Monthly Number of Orders (Aug 2016 - Dec 2016)') 
plt.show()
ax = data_new.set_index('monthordered')['return_rate'].plot(kind = 'bar', color = 'teal')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y))) 

plt.legend()
plt.xlabel('month ordered')
plt.ylabel('return rate')
plt.title('Monthly Return Rate (Aug 2016 - Dec 2016)')
plt.show()
print(len(data['dateordered'].unique()))
pd.to_datetime('2016-12-31') - pd.to_datetime('2016-08-01')
daily_complete = data.loc[data['orderstatus'] == 'complete', ['dateordered', 'orders']]
daily_return = data.loc[data['orderstatus'] == 'returned', ['dateordered', 'orders']].groupby('dateordered').sum().reset_index()
daily = pd.merge(daily_complete, daily_return, on = 'dateordered', how = 'left', suffixes = ('_completed', '_returned'))
del [daily_complete, daily_return]
daily['orders_total'] = daily['orders_completed'] + daily['orders_returned'].fillna(0)
daily['return_rate'] = daily['orders_returned'].fillna(0) / daily['orders_total']
daily['weekday'] = pd.to_datetime(list(daily['dateordered'])).strftime('%a')
daily.set_index('dateordered')['orders_total'].plot(color = 'teal')

plt.legend()
plt.xlabel('date ordered')
plt.ylabel('number of orders')
plt.title('Daily Number of Total Orders (Aug 2016 - Dec 2016)') 
plt.show()
daily = daily.sort_values(by = 'dateordered')

x=daily['dateordered']
y=daily['orders_returned']

fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(1,1,1) 

width = 0.5
idx = np.arange(len(x))
plt.bar(idx, y, width, color='orange',label='')

plt.xlabel('date ordered')
plt.ylabel('number of returns')
plt.title('Daily Number of Returns (Aug 2016 - Dec 2016)') 
plt.xticks((0, 31, 62, 92, 123),('Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
plt.show()
daily.sort_values(by = 'orders_total', ascending = False).head()
daily.sort_values(by = 'orders_returned', ascending = False).head()
ax = daily.set_index('dateordered')['return_rate'].plot(color = 'teal')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y))) 

plt.legend()
plt.xlabel('date ordered')
plt.ylabel('return rate')
plt.title('Daily Return Rate (Aug 2016 - Dec 2016)')
plt.show()
daily.sort_values(by = 'return_rate', ascending = False).head()
weekly = daily.groupby('weekday').mean().reset_index()
weekly['return_rate'] = weekly['orders_returned'] / weekly['orders_total']
weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
weekly['weekday'] = pd.Categorical(weekly['weekday'], categories=weekdays, ordered=True)
weekly = weekly.sort_values(by = 'weekday')
weekly
weekly.set_index('weekday')['orders_total'].plot(kind = 'bar', color = 'teal', label = 'orders_completed')
weekly.set_index('weekday')['orders_returned'].plot(kind = 'bar', color = 'orange')

plt.legend()
plt.xlabel('day of week ordered')
plt.ylabel('average number of orders')
plt.title('Average Number of Orders by Day of Week(Aug 2016 - Dec 2016)') 
plt.show()
ax = weekly.set_index('weekday')['return_rate'].plot(kind = 'bar',color = 'teal')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y))) 

plt.legend()
plt.xlabel('day of week ordered')
plt.ylabel('return rate')
plt.title('Return Rate by Day of Week(Aug 2016 - Dec 2016)')
plt.show()
return_date = data.loc[data['orderstatus'] == 'returned', ['orders', 'datereturned']].groupby('datereturned').sum().reset_index()
return_date['weekday'] = pd.to_datetime(list(return_date['datereturned'])).strftime('%a')
return_date.sort_values('orders', ascending = False).head(10)
return_weekday = return_date.groupby('weekday').sum().reset_index()
return_weekday['weekday'] = pd.Categorical(return_weekday['weekday'], categories=weekdays, ordered=True)
return_weekday = return_weekday.sort_values(by = 'weekday')
return_weekday.set_index('weekday')['orders'].plot(kind = 'bar', color = 'orange')

plt.legend()
plt.xlabel('day of week returned')
plt.ylabel('number of returns')
plt.title('Number of Returns on Date Returned by Day of Week (Aug 2016 - Dec 2016)') 
plt.show()
data['length_return'] = (data['datereturned'] - data['dateordered']).astype('timedelta64[D]')
data['length_return'].describe()
data['length_return'].hist(color = 'teal')
plt.axvline(x = data['length_return'].mean(), color='orange', linestyle='dashed', label = 'mean')
plt.legend()
plt.xlabel('time between order and return (day)')
plt.ylabel('number of returns')
plt.title('Distribution of Time between Order and Return (day)')
plt.show()
data.loc[data['length_return'] > 100, :]
conn = sqlite3.connect('sales.db')

cursor = conn.cursor()
cursor.execute("""CREATE TABLE 'sales_orders'('index' INT, dateordered char(10), datereturned char(10), orderstatus char(8), orders  INT)""")

data.loc[:, ['dateordered', 'datereturned', 'orderstatus', 'orders']].to_sql(name='sales_orders', con=conn,  if_exists='append')
cursor.execute("""SELECT dateordered, SUM(orders) FROM sales_orders WHERE orderstatus = 'complete' 
GROUP BY dateordered ORDER BY SUM(orders) DESC LIMIT 5""")
cnt = cursor.fetchall()
data_df = pd.DataFrame(cnt,columns = ['dateordered','SUM(orders)'])
data_df
cursor.execute("""SELECT orderstatus, SUM(orders) FROM sales_orders GROUP BY orderstatus""")
cnt = cursor.fetchall()
print(cnt)
cursor.execute("""SELECT ROUND(AVG(total_orders), 2) AS daily_avg FROM 
(SELECT dateordered, SUM(orders) AS total_orders FROM sales_orders GROUP BY dateordered)""")
cnt = cursor.fetchall()
print(cnt)