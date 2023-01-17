import numpy as np #linear Algebra

import pandas as pd #Data Exploration 

import seaborn as sns #Data Visualization

import matplotlib.pyplot as plt #Data Visualization - base Library
#show graph outputs

%matplotlib inline 

sns.set_style("whitegrid") #seaborn Styling
df_Orders = pd.read_csv('../input/orders.csv')

df_Prior = pd.read_csv('../input/order_products__prior.csv')

df_Train = pd.read_csv('../input/order_products__train.csv')

df_Products = pd.read_csv('../input/products.csv')

df_aisles = pd.read_csv('../input/aisles.csv')

df_dept = pd.read_csv('../input/departments.csv')
df_Orders.info()
df_Prior.info()
df_Prior_merge  = df_Orders.merge(df_Prior,on='order_id')
df_Prior_merge.head()
df_Prior_merge.nunique()
df_Prior_merge.columns
order_count = df_Prior_merge.groupby('user_id')['order_id'].count()
order_count.sort_values(ascending=False).head()
plt.figure(figsize=(12,4))

sns.set_palette("viridis")

sns.distplot(order_count,kde=False,bins=100)

plt.title("Total no. of Order by each user", fontsize=14)

plt.xlabel('orders', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='order_number',data=df_Prior_merge)

plt.title("Count of each order_number", fontsize=14)

plt.xlabel('order_number', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
df_Prior_merge.columns
dow = df_Prior_merge['order_dow'].value_counts().reset_index()
plt.figure(figsize=(12,4))

sns.barplot(x='index',y='order_dow',data=dow)

plt.title("Total orders on day of week", fontsize=14)

plt.xlabel('day of week', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='order_hour_of_day',data=df_Prior_merge)

plt.title("Total order on hours of day", fontsize=14)

plt.xlabel('hours of day', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.show()
df_time_matrix = df_Prior_merge.groupby(['order_dow','order_hour_of_day']).count().reset_index().pivot('order_hour_of_day','order_dow','order_id')
plt.figure(figsize=(12,6))

sns.heatmap(df_time_matrix,cmap='viridis')

plt.title("Distribution of order over week", fontsize=14)

plt.xlabel('day of week', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.show()
prior_order_freq = df_Prior_merge['days_since_prior_order'].value_counts()
plt.figure(figsize=(13,6))

sns.barplot(x=prior_order_freq.index, y= prior_order_freq.values)

plt.title("Time interval between Orders", fontsize=14)

plt.xlabel('date', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.show()
df_Prior_merge  = df_Prior_merge.merge(df_Products,on='product_id')
df_Prior_merge = df_Prior_merge.merge(df_dept,how='left', on='department_id')
df_Prior_merge = df_Prior_merge.merge(df_aisles,how='left', on='aisle_id')
df_Prior_merge.head()
product_count = df_Prior_merge['product_id'].nunique()

department_count = df_Prior_merge['department_id'].nunique()

asile_count = df_Prior_merge['aisle_id'].nunique()

print("So there are %d products from %d depatments and %d aisle" %(product_count,department_count, asile_count))
df_Prior_merge['product_name'].value_counts().head(10)
plt.figure(figsize=(13,6))

sns.countplot(x='department', data=df_Prior_merge)

plt.title("Department wise Orders", fontsize=14)

plt.xlabel('Department', fontsize=12)

plt.ylabel('counts', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
df_Prior_merge['aisle'].value_counts().head(10)