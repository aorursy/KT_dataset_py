# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

import pandas_profiling

import gc

import datetime

plt.style.use('ggplot')
df=pd.read_csv('../input/ecommerce-data/data.csv',encoding='ISO-8859-1')

df.head()
df.info()
df.isnull().sum().sort_values(ascending=False)
df[df.isna().any(axis=1)].head(10)

#df[df.isnull.any(axis=1)].head(10)
df['InvoiceDate']=pd.to_datetime(df.InvoiceDate,format='%m/%d/%Y %H:%M')
#df.info()
df=df.dropna()
df.isnull().sum().sort_values(ascending=False)
df['CustomerID']=df['CustomerID'].astype('int64')
#df.info()
df2=df.copy()
df2.describe().round(2)
sns.set(style='whitegrid')

ax=sns.violinplot(x=df2['Quantity'])
df2=df2[df2.Quantity>0]

df2.describe().round(2)
df2['AmountSpent']=df2['Quantity']*df2['UnitPrice']

df2.head()
#import datetime

df2['month_year']=pd.to_datetime(df2['InvoiceDate']).dt.to_period('M')

df2.head()
L=['year','month','day','dayofweek','dayofyear','weekofyear','quarter']

df2=df2.join(pd.concat((getattr(df2['InvoiceDate'].dt,i).rename(i) for i in L),axis=1))

df2.head()
df2.dayofweek.unique()
df2['dayofweek']=df2['dayofweek']+1
df2.head()
sales_per_cust=df2.groupby(by=['CustomerID','Country'],as_index=False)['InvoiceNo'].count().sort_values(by='InvoiceNo',ascending=False)

sales_per_cust.columns=['CustomerID','Country','NumberofSales']

sales_per_cust.head()
orders=df.groupby(by=['CustomerID','Country'],as_index=False)['InvoiceNo'].count()

plt.subplots(figsize=(15,6))

plt.plot(orders.CustomerID,orders.InvoiceNo);

plt.xlabel('Customer ID')

plt.ylabel('Number of Orders')

plt.title('Number of Orders for different Customers')

plt.ioff()
orders=df2.groupby(by=['CustomerID','Country'],as_index=False)['AmountSpent'].sum()

plt.subplots(figsize=(15,6))

plt.plot(orders.CustomerID,orders.AmountSpent);

plt.xlabel('Customer ID')

plt.ylabel('Money spent in Dollars')

plt.title('Money spend by different Customers')

plt.ioff()
spent_per_cust=df2.groupby(by=['CustomerID','Country'],as_index=False)['AmountSpent'].sum().sort_values(by='AmountSpent',ascending=False)

#spent_per_cust.columns=['CustomerID','Country','TotalSpent']

spent_per_cust.head(10)
df2.insert(loc=2,column='year_month',value=df2['InvoiceDate'].map(lambda x: 100*x.year + x.month))

df2.insert(loc=5,column='hour',value=df2.InvoiceDate.dt.hour)
ax=df2.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot('bar',color='blue',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for per Month(1st Dec 2010 - 9th Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_10','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'));
df2.groupby('InvoiceNo')['dayofweek'].unique().value_counts().sort_index()
ax=df2.groupby('InvoiceNo')['dayofweek'].unique().value_counts().sort_index().plot('bar',color='blue',figsize=(15,6))

ax.set_xlabel('Day',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Days',fontsize=15)

ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'),rotation='horizontal',fontsize=15);
ax=df2.groupby('InvoiceNo')['hour'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))

ax.set_xlabel('Hour',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Hour',fontsize=15)

ax.set_xticklabels(range(6,21),rotation='horizontal',fontsize=15);

plt.show()
ax=df2.groupby('InvoiceNo')['weekofyear'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))

ax.set_xlabel('Hour',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Hour',fontsize=15)

ax.set_xticklabels(range(0,52),rotation='horizontal',fontsize=15);

plt.show()
df2.UnitPrice.describe()
#plt.subplot(figsize=(12,6))

sns.boxplot(df2.UnitPrice)

plt.show()
df_free=df2[df2.UnitPrice==0]

print(len(df_free))

df_free.head()
df_free.year_month.value_counts().sort_index()
ax=df_free.year_month.value_counts().sort_index().plot('bar',figsize=(12,6),color='blue')

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Frequency',fontsize=15)

ax.set_title('Frequency for different Months (Dec 2010 -Dec 2011)',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','Aug_11','Sep_11','Oct_11','Nov_11'),rotation='horizontal',fontsize=15);

plt.show()
group_country_orders=df2.groupby('Country')['InvoiceDate'].count().sort_values()

plt.subplots(figsize=(15,8))

group_country_orders.plot('barh',fontsize=12,color='blue');

plt.xlabel('Number of orders',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Number of orders of different Countries',fontsize=12)

plt.ioff()
group_country_orders=df2.groupby('Country')['InvoiceDate'].count().sort_values()

group_country_orders_without_uk=group_country_orders.copy()

del group_country_orders_without_uk['United Kingdom']



#plot number of unique customers in each country (without UK)

plt.subplots(figsize=(15,8))

group_country_orders_without_uk.plot('barh',fontsize=12,color='blue');

plt.xlabel('Number of orders',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Number of orders of different Countries without UK',fontsize=12)

plt.ioff()
# Get our date range for our data

print('Date Range: %s to %s' % (df2['InvoiceDate'].min(),df2['InvoiceDate'].max()))



# Since Our data ends at Nov-30 2011 we're taking all the transcations that ocurred before December 01,2011

df2=df2.loc[df2['InvoiceDate']<'2011-12-01']
# Get total amount spent per invoice and associate it with CustomerID and Country

invoice_customer_df=df2.groupby(by=['InvoiceNo','InvoiceDate']).agg({'AmountSpent':sum,'CustomerID':max,'Country':max,}).reset_index()

invoice_customer_df.head()
# Sort on Amount spent,this gives us largest invoices 

invoice_customer_df.sort_values(by='AmountSpent',ascending=False).head(10)
#

#

monthly_repeat_customers_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x: len(x)>1).resample('M').nunique()['CustomerID']

monthly_repeat_customers_df
monthly_unique_customer_df=df2.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()

monthly_unique_customer_df
monthly_repeat_customers_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x:len(x)>1).resample('M').nunique()['CustomerID']

monthly_repeat_customers_df
monthly_repeat_percentage=monthly_repeat_customers_df/monthly_repeat_customers_df*100.0

monthly_repeat_percentage
#Plotting this visully

#Note were using a 2 scale y axis (left and right)



ax=pd.DataFrame(monthly_repeat_customers_df.values).plot(figsize=(12,8))

pd.DataFrame(monthly_unique_customer_df.values).plot(ax=ax,grid=True)



ax2=pd.DataFrame(monthly_repeat_percentage.values).plot.bar(ax=ax,grid=True,secondary_y=True,color='blue',alpha=0.3)



ax.set_xlabel('Date')

ax.set_ylabel('Number of Customers')

ax.set_title('Number of Unique vs. Repeat Customers Over Time')



ax2.set_ylabel('percentage (%)')



ax.legend(['Repeat Customers','All Customers'])

ax2.legend(['Percentage of Repeat'],loc='upper right')



ax.set_ylim([0,monthly_unique_customer_df.values.max()+100])

ax2.set_ylim([0,100])



plt.xticks(range(len(monthly_repeat_customers_df.index)),[x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index],rotation=45)

plt.show()
monthly_revenue_df=df2.set_index('InvoiceDate')['AmountSpent'].resample('M').sum()

monthly_rev_repeat_customer_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x:len(x) > 1).resample('M').sum()['AmountSpent']



monthly_rev_per_repeat_customers_df=monthly_rev_repeat_customer_df/monthly_revenue_df*100

monthly_rev_per_repeat_customers_df

#Plotting this visully

#Note were using a 2 scale y axis (left and right)



ax=pd.DataFrame(monthly_revenue_df.values).plot(figsize=(12,8))

pd.DataFrame(monthly_rev_per_repeat_customers_df.values).plot(ax=ax,grid=True)



#ax2=pd.DataFrame(monthly_repeat_percentage.values).plot.bar(ax=ax,grid=True,secondary_y=True,color='blue',alpha=0.3)



ax.set_xlabel('Date')

ax.set_ylabel('Sales')

ax.set_title('Total Revenue vs. Revenue from Repeat Customer')



ax.legend(['Total Revenue','Repeat Customer Revenue'])

ax.set_ylim([0,max(monthly_revenue_df.values)+100000])

ax2=ax.twinx()



pd.DataFrame(monthly_rev_per_repeat_customers_df.values).plot(ax=ax2,kind='bar',color='blue',alpha=0.3)

ax2.set_ylim([0,max(monthly_rev_per_repeat_customers_df.values)+30])

ax2.set_ylabel('Percentage (%)')

ax2.legend(['Repeat Revenue Percentage'])

ax2.set_xticklabels([x.strftime('%m.%Y') for x in monthly_rev_per_repeat_customers_df.index]);