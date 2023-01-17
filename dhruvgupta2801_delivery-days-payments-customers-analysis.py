# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/olist_sellers_dataset.csv')
df1.tail()
df1['seller_state'].nunique()
df1['seller_city'].nunique()
plt.figure(figsize=(10,5))
sns.barplot(x=df1['seller_state'].value_counts().index,y=df1['seller_state'].value_counts().values)
plt.xlabel('States')
plt.ylabel('No of Sellers')
plt.title('Sellers per State')
z=plt.xticks(rotation=90)
df1['seller_zip_code_prefix'].value_counts().head()
df2=pd.read_csv('../input/olist_orders_dataset.csv')
df2.tail()
X=pd.to_datetime(df2['order_delivered_customer_date'])-pd.to_datetime(df2['order_purchase_timestamp'])
for i in range(0,len(X)):
    X[i]=X[i].days
plt.figure(figsize=(10,5))
sns.barplot(x=X.value_counts().sort_values(ascending=False).head(30).index,y=X.value_counts().sort_values(ascending=False).head(30).values)
plt.xlabel('Deleivery Days')
plt.ylabel('Frequency')
df3=pd.read_csv('../input/olist_order_items_dataset.csv')
df3.head()
df4=pd.read_csv('../input/olist_products_dataset.csv')
df4.head()
z=df3[df3['price']==max(df3['price'])]['product_id'].values.astype(str)
most_expensive_product=[df4['product_id']==z[0]]

df5=pd.read_csv('../input/olist_order_payments_dataset.csv')
df5.head()
sum1=[]
group=[]
for groups,frame in df5.groupby('payment_type'):
    sum1.append(sum(frame['payment_value']))
    group.append(groups)
plt.figure()
sns.barplot(x=group,y=sum1)
plt.xlabel('Payment Type')
plt.ylabel('Total Amount')
plt.title('Payment Type vs Amount Paid')
plt.figure()
sns.barplot(x=df5['payment_type'].value_counts().index,y=df5['payment_type'].value_counts().values)
plt.title('No of Transactions VS Payment Type')
plt.ylabel('No of Transactions')
plt.xlabel('Payment Type')
df6=pd.read_csv('../input/olist_customers_dataset.csv')
df6.head()
plt.figure(figsize=(10,5))
plt.title('Customers Per State')
plt.ylabel('No of Customers')
plt.xlabel('States')
sns.barplot(x=df6['customer_state'].value_counts().index,y=df6['customer_state'].value_counts().values)
df1.head()
S1=pd.Series(data=df6['customer_state'].value_counts().values,index=df6['customer_state'].value_counts().index)
S2=pd.Series(data=df1['seller_state'].value_counts().values,index=df1['seller_state'].value_counts().index)
dfz1=pd.DataFrame(data=S1,index=S1.index,columns=['Customers'])
dfz1['Sellers']=S2
dfz1.dropna().plot.bar(figsize=(15,7))
plt.title('No of Sellers VS Buyers per State')
plt.xlabel('States')
plt.ylabel('Frequency')
dfz1


