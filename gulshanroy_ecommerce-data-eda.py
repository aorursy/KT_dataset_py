# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

ecd = pd.read_csv('../input/ecommerce-data/data.csv', encoding = 'ISO-8859-1')
ecd.head()
ecd.info()
# for null value

ecd.isnull().sum().sort_values(ascending=False)
ecd.describe()
# df_ecd without missing values

df_ecd = ecd.dropna()
df_ecd.isnull().sum().sort_values(ascending=False)
df_ecd.describe()
df_ecd = df_ecd[df_ecd.Quantity > 0]
df_ecd['amount'] = df_ecd['Quantity'] * df_ecd['UnitPrice']
df_ecd.describe()
# change the invoice_date format - String to Timestamp format

df_ecd['InvoiceDate'] = pd.to_datetime(df_ecd.InvoiceDate, format='%m/%d/%Y %H:%M')
df_ecd.insert(loc=6, column='year_month', value=df_ecd['InvoiceDate'].map(lambda x: 100*x.year + x.month))

df_ecd.insert(loc=7, column='month', value=df_ecd.InvoiceDate.dt.month)

# +1 to make Monday=1.....until Sunday=7

df_ecd.insert(loc=8, column='day', value=(df_ecd.InvoiceDate.dt.dayofweek)+1)

df_ecd.insert(loc=9, column='hour', value=df_ecd.InvoiceDate.dt.hour)
df_ecd.head()
import matplotlib.pyplot as plt



orders = df_ecd.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()



plt.subplots(figsize=(15,6))

plt.plot(orders.CustomerID, orders.InvoiceNo)

plt.xlabel('Customers ID')

plt.ylabel('Number of Orders')

plt.title('Number of Orders for different Customers')

plt.show()
print('The TOP 10 customers with most number of orders...')

orders.sort_values(by='InvoiceNo', ascending=False).head(10)
money_spent = df_ecd.groupby(by=['CustomerID','Country'], as_index=False)['amount'].sum()



plt.subplots(figsize=(15,6))

plt.plot(money_spent.CustomerID, money_spent.amount)

plt.xlabel('Customers ID')

plt.ylabel('Money spent')

plt.title('Money Spent for different Customers')

plt.show()
print('The TOP 10 customers with highest money spent...')

money_spent.sort_values(by='amount', ascending=False).head(10)
df_ecd.groupby('InvoiceNo')['month'].unique().value_counts().sort_index()
ax = df_ecd.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot(kind='bar',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()
df_ecd.groupby('InvoiceNo')['day'].unique().value_counts().sort_index()
ax = df_ecd.groupby('InvoiceNo')['day'].unique().value_counts().sort_index().plot(kind ='bar',figsize=(15,6))

ax.set_xlabel('Day',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Days',fontsize=15)

ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'), rotation='horizontal', fontsize=15)

plt.show()
df_ecd.groupby('InvoiceNo')['hour'].unique().value_counts().iloc[:-1].sort_index()
ax = df_ecd.groupby('InvoiceNo')['hour'].unique().value_counts().iloc[:-1].sort_index().plot(kind = 'bar',figsize=(15,6))

ax.set_xlabel('Hour',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Hours',fontsize=15)

ax.set_xticklabels(range(6,21), rotation='horizontal', fontsize=15)

plt.show()
df_ecd.UnitPrice.describe()
# distribution of unit price

plt.subplots(figsize=(12,6))

sns.set(style="whitegrid")

ax = sns.stripplot(x=df_ecd["UnitPrice"])

plt.show()
df_free = df_ecd[df_ecd.UnitPrice == 0]

df_free.head(10)
df_free.year_month.value_counts().sort_index()
ax = df_free.year_month.value_counts().sort_index().plot(kind ='bar',figsize=(12,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Frequency',fontsize=15)

ax.set_title('Frequency for different Months (Dec 2010 - Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11'), rotation='horizontal', fontsize=13)

plt.show()
group_country_orders = df_ecd.groupby('Country')['InvoiceNo'].count().sort_values()

# del group_country_orders['United Kingdom']



# plot number of unique customers in each country (with UK)

plt.subplots(figsize=(15,8))

group_country_orders.plot(kind ='bar', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.ylabel('Number of Orders', fontsize=12)

plt.title('Number of Orders for different Countries', fontsize=12)

plt.show()
group_country_orders = df_ecd.groupby('Country')['InvoiceNo'].count().sort_values()

del group_country_orders['United Kingdom']



# plot number of unique customers in each country (without UK)

plt.subplots(figsize=(15,8))

group_country_orders.plot(kind= 'bar', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.ylabel('Number of Orders', fontsize=12)

plt.title('Number of Orders for different Countries', fontsize=12)

plt.show()
group_country_amount_spent = df_ecd.groupby('Country')['amount'].sum().sort_values()



plt.subplots(figsize=(15,8))

group_country_amount_spent.plot(kind='bar', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.ylabel('Money Spent', fontsize=12)

plt.title('Money Spent by different Countries', fontsize=12)

plt.show()
group_country_amount_spent = df_ecd.groupby('Country')['amount'].sum().sort_values()

del group_country_amount_spent['United Kingdom']



# plot total money spent by each country (without UK)

plt.subplots(figsize=(15,8))

group_country_amount_spent.plot(kind= 'bar', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.ylabel('Money Spent', fontsize=12)

plt.title('Money Spent by different Countries', fontsize=12)

plt.show()