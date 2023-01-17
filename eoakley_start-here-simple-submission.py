import numpy as np

import pandas as pd
orders_prior = pd.read_csv('../input/order_products__prior.csv')

orders_train = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')
order_products = pd.concat([orders_prior, orders_train], axis=0)

del orders_train, orders_prior
def products_concat(vet):

    out = ''

    

    #vet is a pd.Series

    for prod in vet:

        if prod > 0:

            out += str(int(prod)) + ' '

    

    if out != '':

        return out.rstrip()

    else:

        return 'None'
all_data = order_products.merge(orders, on='order_id', how='outer')

del order_products



user_products = all_data.groupby('user_id').product_id.apply(products_concat)
test_set = all_data.loc[all_data.eval_set == 'test'][['order_id', 'user_id']]

test_set = test_set.join(user_products, on='user_id')
submission = pd.DataFrame({'order_id': test_set.order_id, 'products': test_set.product_id})

submission.head()
submission.to_csv('simple_sub.csv', index=False)