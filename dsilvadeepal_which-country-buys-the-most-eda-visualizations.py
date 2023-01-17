#Import Python3 libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#Import data file
df = pd.read_csv('../input/data.csv', encoding = 'ISO-8859-1')
print('Dataframe dimensions:', df.shape)
df.head()
df.describe().round(2)

print(df.dtypes)

#Change the format for the invoice date
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df.head()
#Drop missing values from the dataset
df.dropna(inplace=True)

#Drop duplicates from the dataset
df.drop_duplicates(inplace=True)
# Change customer id type from float to string 
df['CustomerID'] = df['CustomerID'].astype('int')

print(df.dtypes)
print('Dataframe dimensions:', df.shape)
temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()
pd.DataFrame([{'products': len(df['StockCode'].value_counts()),    
               'transactions': len(df['InvoiceNo'].value_counts()),
               'customers': len(df['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], index = ['Counts'])
temp = df.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
temp[ : 5]
temp['canceled_orders'] = temp['InvoiceNo'].apply(lambda x: int('C' in x))
temp = temp.rename(columns = {'InvoiceDate':'No of products'})
temp[ : 5]

num_of_canceled_orders = temp['canceled_orders'].sum()
total_orders = temp.shape[0]
print('Percent of canceled orders: {:.2f}% '.format(num_of_canceled_orders/total_orders * 100))

#Remove line items with negative quantity and unit price
df_new = df[(df.Quantity >= 0) & (df.UnitPrice >= 0)] 
df_new.describe().round(2)
df_new.head()
#Get the total spent for each purchase - Basket price
df_new['TotalDollars'] = df_new['Quantity'] * df_new['UnitPrice']
df_new.head()
orders_by_country = df_new.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)

#Plot
orders_by_country.plot('bar')
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()
spend_by_country = df_new.groupby('Country')['TotalDollars'].mean().sort_values(ascending=False)
#Plot
spend_by_country.plot('bar')
plt.xlabel('Average spend amount in $')
plt.ylabel('Country')
plt.title('Average spend amount per Country', fontsize=16)
plt.show()
orders_by_country = df_new.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)
del orders_by_country['United Kingdom']

#Plot
orders_by_country.plot('bar')
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()