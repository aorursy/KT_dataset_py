# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/Online Retail.xlsx')
df.head()
df=df[['CustomerID','InvoiceNo','StockCode','Quantity','UnitPrice','Description','InvoiceDate','Country']]
# Calculate the total sales of each product
TotalAmount = df['Quantity'] * df['UnitPrice']
df.insert(loc=5,column='TotalAmount',value=TotalAmount)
new_df = df[['CustomerID','InvoiceNo','StockCode','Quantity','TotalAmount','InvoiceDate','Country']]
# Show top 10 customers, max and min amounts paid   
res_df = new_df.groupby(['CustomerID']).sum()
res_df.sort_values('TotalAmount',ascending=False,inplace=True)
final_df = res_df[(res_df['Quantity'] > 0) & (res_df['TotalAmount'] > 0)]

print('customer Id {} who paid the maximum amount {}'.format(int(final_df['TotalAmount'].argmax()),final_df['TotalAmount'].max()))
print('customer Id {} who paid the minimum amount {}'.format(int(final_df['TotalAmount'].argmin()),final_df['TotalAmount'].min()))

final_df.head(10)
# total sales at each country, how many quantites sold at each country 
country_df = new_df.groupby(['Country']).sum()
country_df.sort_values('TotalAmount',ascending=False,inplace=True)
country_df.drop('CustomerID',axis=1,inplace=True)
country_df.head()
# show the min and max quantites has sold. 
print('The minimum number of products has been bought is {} by customer id {} '.format(final_df['Quantity'].min(),final_df['Quantity'].argmin()))
print('The maximum number of products has been bought is {} by customer id {} '.format(final_df['Quantity'].max(),final_df['Quantity'].argmax()))
new_df.groupby('Country').mean()
avg_quan = new_df[['Quantity','TotalAmount','Country','InvoiceNo','CustomerID']]
## Top 10 customers sales overall countries sorted by totalamount
avg_sum = avg_quan.groupby(['Country','CustomerID']).sum() 
avg_sum.sort_values('TotalAmount',ascending=False).head(10)
## average of paid amount by each customer at each country ordered by number of invoices.
avg_cus = df[['Quantity','TotalAmount','Country','InvoiceNo']].copy()

x = avg_cus.groupby(['Country','InvoiceNo']).sum()

x['Ones']=1
y = x.groupby('Country').sum()
y['AVG'] = y['TotalAmount'] / y['Ones']
y.sort_values(['Ones','AVG'],ascending=False,inplace=True)
y.head()
## Average of amount paid by the customers overall countries
y['TotalAmount'].sum() / y['Quantity'].sum()
y['AVG'].plot(kind='bar',figsize=(10,5),title='Average amount paid by the customer over all countries')
plt.ylabel('AVG')
plt.xlabel('Country')
modifed_df = new_df[['Quantity','TotalAmount','InvoiceDate','Country']].copy()
modifed_df['Month'] = modifed_df['InvoiceDate'].dt.month 
modifed_df['Year'] = modifed_df['InvoiceDate'].dt.year 
date_df = modifed_df.groupby(['Year','Month']).sum()
total_values = date_df.sort_values('TotalAmount',ascending=False)
total_values
total_values.plot(kind='bar',figsize=(10,5),title='Graph show total sales and quatities at each month')
plt.ylabel('Quantity, total Amount')
# I showed which month at each country has the higest total sales    
country_df = modifed_df.groupby('Country').max()
country_df.sort_values('TotalAmount',ascending=False,inplace=True)
country_df
## Total sales for each product

df_1=df[['StockCode','Description','Quantity','TotalAmount','Country','InvoiceDate']]
product_totalsales_df = df_1.groupby(['Country','StockCode','Description']).sum()
product_totalsales_df = product_totalsales_df[ (product_totalsales_df['Quantity'] > 0) & (product_totalsales_df['TotalAmount'] >0) ]
product_totalsales_df = product_totalsales_df[product_totalsales_df['TotalAmount'] >= 1000 ]

product_totalsales_df.sort_values('TotalAmount',ascending=False)
## Sales Average of each product overall countries 

Avgsales_product = df_1.groupby(['StockCode','Description']).mean()

Avgsales_product = Avgsales_product[ (Avgsales_product['Quantity'] > 0) & (Avgsales_product['TotalAmount'] >0) ]
Avgsales_product = Avgsales_product[Avgsales_product['TotalAmount'] >= 100 ]

Avgsales_product.sort_values('TotalAmount',ascending=False)

## total sales for each product at each country

Total_sales_product = df_1.groupby(['Country','StockCode']).sum()

Total_sales_product = Total_sales_product[ (Total_sales_product['Quantity'] > 0) & (Total_sales_product['TotalAmount'] >0) ]
Total_sales_product = Total_sales_product[Total_sales_product['TotalAmount'] >= 100 ]

Total_sales_product
df_1['Month'] = df_1['InvoiceDate'].dt.month.copy()
## total sales of product at each month at each country I made filter on total sales >= 100

df_1 = df_1[ (df_1['Quantity'] > 0) & (df_1['TotalAmount'] >0) ]
df_1 = df_1[df_1['TotalAmount'] >= 100 ]

df_1.groupby(['Country','Month','StockCode','Description']).sum()

