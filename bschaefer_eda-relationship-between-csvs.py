import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_path = '../input'



df_aisles = pd.read_csv(data_path + '/aisles.csv')

df_departments = pd.read_csv(data_path + '/departments.csv')

df_products = pd.read_csv(data_path + '/products.csv')



df_orders = pd.read_csv(data_path + '/orders.csv')

df_ord_prod_train = pd.read_csv(data_path + '/order_products__train.csv')

df_ord_prod_prior = pd.read_csv(data_path + '/order_products__prior.csv')
print("df_orders shape: {}".format(df_orders.shape))

print(df_orders.groupby('eval_set').size())

df_orders.head(15)
df_orders[df_orders.user_id <= 10].groupby(['user_id','eval_set']).size().unstack(fill_value=0)
plt.figure(figsize=(20,8))

ax = sns.countplot(df_orders['user_id'].value_counts())
print("df_ord_prod_train shape: {}".format(df_ord_prod_train.shape))

print("df_ord_prod_prior shape: {}".format(df_ord_prod_prior.shape))
df_ord_prod_prior.head(3)
df_ord_prod_prior['eval_set'] = 'prior'

df_ord_prod_train['eval_set'] = 'train'

df_order_products = df_ord_prod_prior.append(df_ord_prod_train, ignore_index=True)
cnt_products_per_order = df_order_products.groupby('order_id').size()

plt.figure(figsize=(20,8))

sns.countplot(cnt_products_per_order)

xt = plt.xticks(rotation='vertical')
print("df_aisles shape: {}".format(df_aisles.shape))

print("df_departments shape: {}".format(df_departments.shape))

print("df_products shape: {}".format(df_products.shape))
df_products = df_products.merge(df_aisles).merge(df_departments)

df_products.head()