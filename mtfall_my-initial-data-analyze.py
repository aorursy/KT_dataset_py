import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

order_products__prior = pd.read_csv('../input/order_products__prior.csv')

order_products__train = pd.read_csv('../input/order_products__train.csv')

order = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
display(order)

print(order.shape)
order.groupby('eval_set').size()
order.groupby('user_id').size().hist()

print(order.groupby('user_id').size().mean())
order_products__prior
order_products__prior.groupby('order_id').size().hist()

print(order_products__prior.groupby('order_id').size().mean())
order_products__train.head()
display(products)

print(products.shape)
display(aisles.head())

print(aisles.shape)

display(departments.head())

print(departments.shape)
display(sample_submission.head())

print(sample_submission.shape)

display(sample_submission.ix[0,'products'])