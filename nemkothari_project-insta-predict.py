# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

order_products__prior = pd.read_csv('../input/order_products__prior.csv')

order_products__train = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import seaborn as sns

plt.style.use('ggplot')

from sklearn.metrics import roc_auc_score,accuracy_score,f1_score

from sklearn.metrics import confusion_matrix

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
print(aisles.info())

aisles.head()
print(departments.info())

departments.head()
plt.hist('order_dow',data=orders,  bins=[0,1,2,3,4,5,6,7])

plt.xlabel('Day of Week')

plt.ylabel('Count')

plt.title('Orders by Day of Week')
plt.hist('order_hour_of_day',data=orders, bins=np.arange(0,24))

plt.xlabel('Hour of Day')

plt.ylabel('Count')

plt.title('Orders by Hour of Day')
grouped = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].count().reset_index()

grouped = grouped.pivot('order_hour_of_day', 'order_dow', 'order_number')

sns.heatmap(grouped)
plt.hist('days_since_prior_order',data=orders.dropna(),bins=np.arange(0,31)) #NaNs are dropped

plt.xlabel('Days Since Prior Order')

plt.ylabel('Count')

plt.title('Days Since Prior Order')
temp_merged = pd.merge(products,aisles,on='aisle_id')

products_merged = pd.merge(temp_merged,departments,on='department_id')

merged_order_products__prior = pd.merge(order_products__prior, products_merged, on='product_id',how='left')

merged_order_products__prior.head()
product_counts = merged_order_products__prior['product_name'].value_counts()

product_counts.head(20)
aisle_counts = merged_order_products__prior['aisle'].value_counts().head(20)

sns.barplot(aisle_counts.index,aisle_counts.values,color='green')

plt.xticks(rotation=90)
department_counts = merged_order_products__prior['department'].value_counts()

sns.barplot(department_counts.index,department_counts.values,color='blue')

plt.xticks(rotation=90)
products_in_order = merged_order_products__prior.groupby('order_id')['add_to_cart_order'].max()

plt.hist(products_in_order,bins=np.arange(1,50))
products_reorder = merged_order_products__prior.groupby('product_name')['reordered'].mean().sort_values(ascending=False)

products_reorder = products_reorder.head(20)

sns.barplot(products_reorder.index,products_reorder.values,color='aqua')

plt.xticks(rotation='vertical')

plt.ylabel('Reorder Rate')
prev_orders = orders.groupby('user_id')['order_number'].max().value_counts()

plt.figure(figsize=(13,6))

sns.barplot(prev_orders.index,prev_orders.values)

plt.xticks(rotation='vertical')

plt.xlabel('Number of Orders')

plt.ylabel('Number of Customers')
merged_order_products__prior.head()
merged = pd.merge(orders,merged_order_products__prior,on='order_id',how='right')
merged.head()
users = pd.DataFrame()

users['average_days_in_between'] = orders.groupby('user_id')['days_since_prior_order'].mean() 

users['number_of_orders_users'] = orders.groupby('user_id').size()

users['total_items'] = merged.groupby('user_id').size()

users['all_products'] = merged.groupby('user_id')['product_id'].apply(set)

users['total_distinct_items'] = users.all_products.map(len)

users['average_basket'] = users.total_items / users.number_of_orders_users

users=users.reset_index()

users=users.set_index('user_id',drop=False)

print(users.shape)

users.head()