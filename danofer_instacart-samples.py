import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
# import lightgbm as lgb

IDIR = '../input/'
print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
#         usecols=['product_id', 'aisle_id', 'department_id']
                      )

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))
print("Original train mean reorder [no grouping]:", train.reordered.mean())
print("Original train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Original train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())
orders.nunique()
orders["user_count"] = orders.groupby("user_id")["order_id"].transform("count")
# orders.user_count.describe()
orders.user_count.hist()
orders = orders.loc[(orders.user_count > 10) ] #  & (orders.user_count <96)
orders.shape
orders.user_count.hist()
orders.nunique()
import random

sample_users = random.sample(set(orders.user_id),50123)
len(sample_users)                 
orders = orders.loc[orders.user_id.isin(sample_users)]
orders.shape
orders.head()
train.head()
train.shape
train.loc[train.order_id.isin(orders.order_id)].shape
train = train.loc[train.order_id.isin(orders.order_id)]
print(train.shape)
train["product_count"] = train.groupby("product_id")["reordered"].transform("count")
train["product_count"].hist()
train["product_count"].describe()
print("Group filt train mean reorder [no grouping]:", train.reordered.mean())
print("Group filt train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Group filt train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())
train = train.loc[train["product_count"]>2]
print(train.shape)
train["product_count"].describe()
products.shape
print("Original train mean reorder [no grouping]:", train.reordered.mean())
print("Original train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Original train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())
train["product_count"] = train.groupby("product_id")["reordered"].transform("count")
orders["user_count"] = orders.groupby("user_id")["order_id"].transform("count")
df_prior = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
print(df_prior.shape)
df_prior.head()
df_prior = df_prior.loc[df_prior.order_id.isin(orders.order_id)]
print(df_prior.shape)
df_prior.reordered.mean()
## Filter by items also
df_prior = df_prior.loc[df_prior.product_id.isin(train.product_id)]
print(df_prior.shape)
df_prior.head()
df_prior.reordered.mean().round(2)
df_prod = pd.read_csv(IDIR + 'products.csv')
print(df_prod.shape)

df_prod = df_prod.merge(pd.read_csv(IDIR + 'aisles.csv'),on="aisle_id")
df_prod = df_prod.merge(pd.read_csv(IDIR + 'departments.csv'),on="department_id")

print(df_prod.shape)
df_prod.head()
train.to_csv("instacart_train_sample_50k.csv.gz",index=False,compression="gzip")
orders.to_csv("instacart_orders_sample_50k.csv.gz",index=False,compression="gzip")
df_prior.to_csv("instacart_priorOrders_sample_50k.csv.gz",index=False,compression="gzip")

df_prod.to_csv("products.csv.gz",index=False,compression="gzip")
