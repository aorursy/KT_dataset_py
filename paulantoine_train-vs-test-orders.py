import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



IDIR = '../input/'

orders = pd.read_csv(IDIR + 'orders.csv')



test_orders = orders[orders.eval_set == 'test']

train_orders = orders[orders.eval_set == 'train']



test_users = test_orders.user_id.values

train_users = train_orders.user_id.values

nb_test_users = len(test_users)

nb_train_users = len(train_users)



orders_by_test_users = orders[orders.user_id.isin(test_users)]

orders_by_train_users = orders[orders.user_id.isin(train_users)]



nb_test_orders = orders_by_test_users.shape[0]

nb_train_orders = orders_by_train_users.shape[0]



if len(set(train_users) & set(test_users)) == 0:

    print("No overlap between users from train and test set\n")

    

print('test: {} users, {} orders, {} orders per user'.format(nb_test_users,

                                                             nb_test_orders,

                                                             nb_test_orders / nb_test_users))

print('train: {} users, {} orders, {} orders per user'.format(nb_train_users,

                                                             nb_train_orders,

                                                             nb_train_orders / nb_train_users))
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={

            'order_id': np.int32,

            'product_id': np.uint16,

            'add_to_cart_order': np.int16,

            'reordered': np.int8})



# add order info to priors

orders.set_index('order_id', inplace=True, drop=False)

priors = priors.join(orders, on='order_id', rsuffix='_')

priors.drop('order_id_', inplace=True, axis=1)



# build users details dataframe

usr = pd.DataFrame()

usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)

usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)



users = pd.DataFrame()

users['total_items'] = priors.groupby('user_id').size().astype(np.int16)

users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)

users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)



users = users.join(usr)

del usr

users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)

print('user f', users.shape)
users.loc[train_users].describe()
users.loc[test_users].describe()