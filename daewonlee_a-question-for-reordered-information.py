# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

aisles = pd.read_csv("../input/aisles.csv")

departments = pd.read_csv("../input/departments.csv")

products = pd.read_csv("../input/products.csv")

orders = pd.read_csv("../input/orders.csv")

train = pd.read_csv("../input/order_products__train.csv")

prior = pd.read_csv("../input/order_products__prior.csv")
orders_1 = orders[orders['user_id'] == 1]  # The orders of user_id=1 in the prior.

orders_1 = orders_1.sort_values('days_since_prior_order')

orders_1 = pd.merge(orders_1, prior, on='order_id')

orders_1 = pd.merge(orders_1, products, on='product_id')

orders_1.head(10)
df = orders_1[orders_1['product_name'] == 'Zero Calorie Cola']

df[['user_id', 'days_since_prior_order', 'product_name', 'reordered']]
df = orders_1[orders_1['product_name'] == 'Organic String Cheese']

df[['user_id', 'days_since_prior_order', 'product_name', 'reordered']]