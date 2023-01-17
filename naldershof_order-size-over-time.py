import pandas as pd

%matplotlib inline
orders = pd.read_csv('../input/orders.csv')

order_products_train = pd.read_csv('../input/order_products__train.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')
order_prod = pd.concat([order_products_train, order_products_prior])
# Arbitrary selection for memory constraints

orders = orders[orders.order_number < 20]
full_order = orders.merge(order_prod, on='order_id')

order_items = full_order[['order_id',

                          'product_id',

                          'order_number']].groupby(['order_id',

                                                    'order_number']).count()



count_by_number = order_items.reset_index()

prod_count = count_by_number[['order_number', 

                              'product_id']].groupby('order_number').mean().plot(ylim=[0,15])