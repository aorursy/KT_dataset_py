import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
jan = pd.read_csv('../input/monthly-sales-data/Sales_April_2019.csv')
feb = pd.read_csv('../input/monthly-sales-data/Sales_February_2019.csv')
mar = pd.read_csv('../input/monthly-sales-data/Sales_March_2019.csv')
apr = pd.read_csv('../input/monthly-sales-data/Sales_April_2019.csv')
may = pd.read_csv('../input/monthly-sales-data/Sales_May_2019.csv')
jun = pd.read_csv('../input/monthly-sales-data/Sales_June_2019.csv')
jul = pd.read_csv('../input/monthly-sales-data/Sales_July_2019.csv')
aug = pd.read_csv('../input/monthly-sales-data/Sales_August_2019.csv')
sep = pd.read_csv('../input/monthly-sales-data/Sales_September_2019.csv')
oct = pd.read_csv('../input/monthly-sales-data/Sales_October_2019.csv')
nov = pd.read_csv('../input/monthly-sales-data/Sales_November_2019.csv')
dec = pd.read_csv('../input/monthly-sales-data/Sales_December_2019.csv')
jan.head()
sales=pd.concat([jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec])
sales.head()
sales.isnull().sum()
sales.dropna(subset = ['Order ID'], inplace=True)
sales.isnull().sum()
sales.info()
# Adding a month column
sales['Month'] = sales['Order Date'].str[:2]
sales = sales.loc[sales['Month']!= 'Or'] # to overcome the error invalid literal for int() with base 10: 'Or'
sales['Month'] = sales['Month'].astype('int32')
sales['Month Name'] = sales['Month'].apply(lambda x: calendar.month_name[x])
# Convert column to correct type
sales['Price Each'] = sales['Price Each'].astype('float')
sales.info()
sales['Quantity Ordered'] = sales['Quantity Ordered'].astype('int32') #Convert column to correct type
sales['Total Sales'] = sales['Price Each']*sales['Quantity Ordered']
sales.head()
per_month = sales.groupby(['Month Name','Month'])
# Making a Dataframe with total sales per month
sales_per_month = per_month['Total Sales'].sum().reset_index()
sales_per_month.sort_values('Month', inplace=True)
plt.figure(figsize=(6,4))
plt.bar(sales_per_month['Month Name'],sales_per_month['Total Sales'])
plt.xticks(rotation='vertical')
plt.xlabel('Month')
plt.ylabel('Sales per Month in $')
plt.title('Sales distribution per month')
plt.show()
# Extracting the City from the address
sales['Purchase Address'] = sales['Purchase Address'].str.split(',')
sales['City'] = sales['Purchase Address'].apply(lambda x: x[1])
sales.head()
# Putting the Address field to as it was before, see the difference in the values covered with [] 
sales['Purchase Address'] = sales['Purchase Address'].str.join(',')
sales.head()
per_city = sales.groupby(['City'])
# Making a Dataframe with total sales per month
sales_per_city = per_city['Total Sales'].sum().reset_index()

plt.figure(figsize=(6,4))
plt.bar(sales_per_city['City'],sales_per_city['Total Sales'])
plt.xticks(rotation='vertical')
plt.xlabel('City')
plt.ylabel('Sales per City in $')
plt.title('Sales distribution per city in the US')
plt.show()
sales.head()
#Converting Order Date Column from String to Date-time data type
sales['Order Date'] = pd.to_datetime(sales['Order Date'])
sales.info()   #Notice the change in Order Date type
sales.head()   
sales['Order Hour'] = sales['Order Date'].dt.hour
sales['Order Minute'] = sales['Order Date'].dt.minute
sales.head()
per_hour = sales.groupby(['Order Hour'])
# Making a Dataframe with total sales per month
sales_per_hour = per_hour['Total Sales'].sum().reset_index()

plt.figure(figsize=(8,4))
plt.plot(sales_per_hour['Order Hour'],sales_per_hour['Total Sales'])
plt.grid()
plt.xlabel('Hour')
plt.xticks(sales_per_hour['Order Hour'])
plt.ylabel('Sales per hour in $')
plt.title('Sales distribution throughout the day')
plt.show()
per_hour_per_city = sales.groupby(['City','Order Hour'])
# Making a Dataframe with total sales per month
sales_per_hour = per_hour_per_city['Total Sales'].sum().reset_index()
sales_per_hour.head()
sales_per_hour['City'].unique()
plt.figure(figsize=(18,12))

plt.subplot(331)
plt.title('Atlanta')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Atlanta']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Atlanta']
plt.plot(x,y)

plt.subplot(332)
plt.title('Austin')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Austin']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Austin']
plt.plot(x,y)

plt.subplot(333)
plt.title('Boston')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Boston']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Boston']
plt.plot(x,y)

plt.subplot(334)
plt.title('Dallas')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Dallas']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Dallas']
plt.plot(x,y)

plt.subplot(335)
plt.title('Los Angeles')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Los Angeles']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Los Angeles']
plt.plot(x,y)

plt.subplot(336)
plt.title('New York City')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' New York City']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' New York City']
plt.plot(x,y)

plt.subplot(337)
plt.title('Portland')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Portland']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Portland']
plt.plot(x,y)

plt.subplot(338)
plt.title('San Francisco')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' San Francisco']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' San Francisco']
plt.plot(x,y)

plt.subplot(339)
plt.title('Seattle')
x= sales_per_hour['Order Hour'].loc[sales_per_hour['City']== ' Seattle']
y= sales_per_hour['Total Sales'].loc[sales_per_hour['City']== ' Seattle']
plt.plot(x,y)
plt.show()
sales.head()
df = sales[sales['Order ID'].duplicated(keep=False)]
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df.head()
df = df[['Order ID','Grouped']].drop_duplicates()
df.head()
from itertools import combinations
from collections import Counter

count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

for key,value in count.most_common(10):
    print(key, value)
prod_group = sales.groupby('Product')
quantity_per_prod = prod_group['Quantity Ordered'].sum().reset_index()
price_per_prod = prod_group['Price Each'].mean().reset_index()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(quantity_per_prod['Product'],quantity_per_prod['Quantity Ordered'], color='green')
ax2.plot(price_per_prod['Product'],price_per_prod['Price Each'],'r-')
ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered',color='green')
ax2.set_ylabel('Price in $',color='red')
ax1.set_xticklabels(quantity_per_prod['Product'],rotation='vertical')
plt.title('Price and Quantity Sold')
plt.show()