import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

priors = pd.read_csv("../input/order_products__prior.csv")

train = pd.read_csv("../input/order_products__train.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

aisles = pd.read_csv("../input/aisles.csv")

departments = pd.read_csv("../input/departments.csv")
priors.head()
order_prod_cnt = priors.groupby('order_id')['reordered'].agg({'prod_cnt':'count', 'reorder_cnt':'sum'}).reset_index()

order_prod_cnt.head()
order_none = order_prod_cnt[order_prod_cnt.reorder_cnt == 0]

order_none['product_id'] = -1

order_none = order_none.drop(['prod_cnt','reorder_cnt'],axis=1)

order_none.head()
priors_new = pd.concat([priors[:3],order_none[:3]])

priors_new['reordered'] = priors_new['reordered'].fillna(1)

priors_new['add_to_cart_order'] = priors_new['add_to_cart_order'].fillna(-1)

priors_new.head(6)
orders_prodstats = orders.merge(right=order_prod_cnt, how='inner', on='order_id')

orders_prodstats.head()
orders.head()
orders.loc[:,'days_T0'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()

orders.head()
print(train.head())

print(products.head())

print(aisles.head())

print(departments.head())
print('aisle number:      ', aisles.shape[0])

print('department number: ', departments.shape[0])

print('product id number: ', products.shape[0])
priors_orders = orders_prodstats.merge(right=priors, how='inner', on='order_id')

#priors_orders = priors_orders.merge(right=products[['product_id','aisle_id','department_id']], how='inner', on='product_id')

priors_orders.head()