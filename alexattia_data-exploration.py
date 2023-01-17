import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import calendar

%matplotlib inline
# departements and aisle id

aisles = pd.read_csv('../input/aisles.csv')

departements = pd.read_csv('../input/departments.csv')

products = pd.read_csv('../input/products.csv')

product_mapping = products[['product_id','product_name']].set_index('product_id').to_dict()['product_name']

print(aisles.head(3))

print(departements.head(3))

print(products.head(3))
# Products purchased in each order. 

# Order_products__prior.csv : contains previous order contents for all customers. 

# 'reordered' customer has a previous order with this product

order_products = pd.read_csv('../input/order_products__prior.csv')

order_products.head(3)
# Order with sets (prior, train, test). 

# You are predicting reordered items only for the test set orders. 

# 'order_dow' is the day of week

orders = pd.read_csv('../input/orders.csv')

orders.sample(3)
f, ax = plt.subplots(figsize=(16,5), ncols=2)

top_k = 15

order_products.product_id.value_counts()[:top_k].plot('bar', ax=ax[0])

_ = ax[0].set_title('%s most ordered products' % top_k)

_ = ax[0].set_xlabel('Products')

_ = ax[0].set_ylabel('Count')

_ = ax[0].set_xticklabels([product_mapping[int(k.get_text())] for k in ax[0].get_xticklabels()])



prod_id_col = order_products.groupby('order_id').count()['product_id']

prod_id_col.value_counts()[:top_k].plot('bar', ax=ax[1])

_ = ax[1].set_title('Repartition of the number products per order')

_ = ax[1].set_xlabel('Number of products per order')

_ = ax[1].set_ylabel('Count')

print('Mean number of products per order : %d' % prod_id_col.mean())
f, ax = plt.subplots(figsize=(16,5), ncols=2)

top_k = 15

orders.order_dow.value_counts().sort_index().plot('bar', ax=ax[0])

_ = ax[0].set_title('Day with the most orders')

_ = ax[0].set_ylabel('Count')

_ = ax[0].set_xticklabels([calendar.day_name[int(k.get_text())] for k in ax[0].get_xticklabels()])



orders.order_hour_of_day.value_counts().sort_index().plot('bar', ax=ax[1])

_ = ax[1].set_title('Hours of the day with the most orders')

_ = ax[1].set_xlabel('Hour of the day')

_ = ax[1].set_ylabel('Count')
f, ax = plt.subplots(figsize=(18,6))

d = orders.pivot_table(index='order_dow', columns='order_hour_of_day', values='order_id', aggfunc=lambda x:round(len(x)*0.001,2)).fillna(0)

f = sns.heatmap(d, annot=True, fmt="1.1f", linewidths=.5, ax=ax) 

_ = ax.set_yticklabels([calendar.day_name[int(k.get_text())][:3] for k in ax.get_yticklabels()])

_ = ax.set_ylabel('')

_ = ax.set_title('Distribution of the numbers of orders')

_ = ax.set_xlabel('Hour of the day')

f, ax = plt.subplots(figsize=(10,10), nrows=2)

_ = ax[0].boxplot(list(orders.days_since_prior_order.dropna()), 0, 'rs', 0)

_ = ax[0].set_xlabel('Days since prior order')

_ = ax[0].set_title('Distribution of the number of days since prior order')



orders.days_since_prior_order.value_counts().sort_index().plot('bar', ax=ax[1])

_ = ax[1].set_xlabel('Days since prior order')

_ = ax[1].set_ylabel('Count')