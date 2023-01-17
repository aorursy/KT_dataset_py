import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

%matplotlib inline
aisles=pd.read_csv('../input/aisles.csv')

aisles.head()
aisles.shape
dept=pd.read_csv('../input/departments.csv')

dept.head()
dept.shape
product=pd.read_csv('../input/products.csv')

product.head()
product.shape
orders=pd.read_csv('../input/orders.csv')

orders.head()
print(orders.groupby('user_id').size().mean())
products_train=pd.read_csv('../input/order_products__train.csv')

products_train.head()
products_prior=pd.read_csv('../input/order_products__prior.csv')

products_prior.head()
orders.groupby('order_dow')['order_id'].count().reset_index().sort_values('order_id', ascending=False).plot(kind='bar')

orders.groupby('order_hour_of_day')['order_id'].count().reset_index().sort_values('order_hour_of_day', ascending=False).plot(kind='bar')
sns.countplot(x='order_dow', data=orders)

plt.xlabel('Day of week')

plt.ylabel('count')

plt.title('Frequency of order by weekday')
sns.countplot(x='order_hour_of_day', data=orders)

plt.xlabel('Hour of day')

plt.ylabel('count')

plt.title('Frequency of order by hour')
plt.figure(figsize=(10,8))

sns.countplot(x='days_since_prior_order', data=orders)

plt.xlabel('Days since prior order')

plt.ylabel('count')

plt.title('Frequency by days since prior order')
df=pd.merge(left=pd.merge(left=product, right=aisles, how='left'),right=dept,how='left')

df.head()
df.groupby(['department']).count()['product_id'].plot(kind='bar', title='number of product in department')
df.describe()
best_selling=products_train['product_id'].value_counts().to_frame()

best_selling['count']=best_selling.product_id

best_selling['product_id']=best_selling.index

merge_df=pd.merge(best_selling, product,how='left',on='product_id').sort_values(by='count', ascending=False)
merge_df.head(15)
# Initialize the matplotlib figure

f,ax=plt.subplots(figsize=(18,10))

sns.barplot(x='product_name',y='count',data=merge_df.head(15),label="best selling products")