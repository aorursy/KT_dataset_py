# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
aisles = pd.read_csv('../input/aisles.csv')
aisles = aisles.sort_values('aisle_id')
aisles
products = pd.read_csv('../input/products.csv')
products = products.sort_values('aisle_id')
products

#table = products.groupby('aisle_id')[products['aisle_id'] == aisles['aisle_id']]
#table
#table = products.join(aisles, lsuffix='_aisles1', rsuffix='_aisles2')
#table
#table = pd.concat([products, aisles])
#table
product_with_aisle = pd.merge(products, aisles, on='aisle_id')
product_with_aisle
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
order_products_prior = products.sort_values('aisle_id')
product_aisle_order = pd.merge(product_with_aisle, order_products_prior, on='product_id')
product_aisle_order = product_aisle_order[['product_id', 'product_name_x', 'aisle_id_x', 'department_id_x', 'aisle']]
table2_columns_rename = {
    "product_name_x": "product_name",
    "aisle_id_x": "aisle_id",
    "department_id_x": "department_id"
}
product_aisle_order.rename(inplace=True, columns=table2_columns_rename)
product_aisle_order
aisle_table = product_aisle_order[['aisle_id', 'aisle']]
aisle_table = aisle_table.groupby('aisle')[['aisle_id']].count().sort_values(['aisle_id'], ascending=False)
aisle_table.head(10)
aisle_table.head(10).plot(kind = 'barh').invert_yaxis()
aisle_table.tail(10).plot(kind = 'barh')
orders = pd.read_csv('../input/orders.csv')
orders = orders[(orders.eval_set == 'prior')]
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
order_and_prior = pd.merge(orders, order_products_prior, on="order_id")
order_and_prior
order_and_prior_and_product = pd.merge(products, order_and_prior, on="product_id")
order_and_prior_and_product
