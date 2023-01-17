import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline
aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

order_products_train = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')
order_products_train.head()
order_products_train.reordered.hist()
most_ordered = order_products_train[['product_id',

                                     'order_id']].groupby('product_id').count().sort_values(ascending=False,

                                                                                            by='order_id')
most_ordered.merge(products, left_index=True, right_on='product_id')
prod_df = order_products_train.merge(products, on='product_id').merge(aisles, on='aisle_id').merge(departments, on='department_id')

g1 = prod_df[['aisle','order_id']].groupby('aisle').count().sort_values('order_id', ascending=False).head(10)

g2 = prod_df[['department','order_id']].groupby('department').count().sort_values('order_id', ascending=False).head(10)
g1.plot(kind='bar')
g2.plot(kind='bar')