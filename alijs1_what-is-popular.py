import numpy as np

import pandas as pd



prior = pd.read_csv('../input/order_products__prior.csv')

print('Got prior products: {}'.format(prior.shape))

train = pd.read_csv('../input/order_products__train.csv')

print('Got train products: {}'.format(train.shape))

all_orders = pd.concat([prior, train])

print('All together: {}'.format(all_orders.shape))
#Let's see, what's popular...

all_order_products = all_orders.loc[all_orders['reordered'] == 1, 'product_id']

products_dict = pd.read_csv('../input/products.csv')

print('Top reordered products are...')

product_counts = all_order_products.value_counts().reset_index()

product_counts.columns = ['product_id','reordered']

product_counts['share'] = product_counts['reordered'] / product_counts['reordered'].sum()

product_counts = product_counts.merge(products_dict[['product_id','product_name']], how='left', on='product_id')

product_counts.head(30)
orders = pd.read_csv('../input/orders.csv')

print('Got orders: {}'.format(orders.shape))

user_order_products = all_orders.loc[all_orders['reordered'] == 1, ['product_id','order_id']]

user_order_products = user_order_products.merge(orders[['order_id','user_id']], how='left', on='order_id')

products_by_user = user_order_products.groupby(['user_id','product_id']).count().reset_index()

products_by_user.columns = ['user_id','product_id','reordered']

products_by_user = products_by_user.merge(products_dict[['product_id','product_name']], how='left', on='product_id')

print('Users with most products reordered')

products_by_user.sort_values('reordered', ascending=False)[['user_id','reordered','product_name']].head(50)
treshold = 2 # Only look at products, which have been re-ordered more that this

max_products = 5 # Max count of products per user

products_by_user = products_by_user.loc[products_by_user['reordered'] > treshold]

print('Shape of products_by_user: {}'.format(products_by_user.shape))

def concat_products(group):

    return " ".join(group[['product_id','reordered']].sort_values('reordered', ascending=False)['product_id'].astype(str).tolist()[:max_products])

products_by_user = products_by_user.groupby('user_id', sort=False).apply(concat_products).reset_index()

products_by_user.columns = ['user_id','products']

products_by_user

sample = pd.read_csv('../input/sample_submission.csv')

submission = sample.merge(orders, how='left', on='order_id')[['order_id','user_id']]

submission = submission.merge(products_by_user, how='left', on='user_id')[['order_id','products']].fillna('None')

submission.to_csv('simple_btb_1.csv', index=False)

submission