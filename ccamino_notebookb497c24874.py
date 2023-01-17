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
# Other

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting
# Dataframes

#Now let's get and put the data in  pandas dataframe



order_products_train = pd.read_csv('../input/order_products__train.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')
print("The order_products_train size is : ", order_products_train.shape)

print("The order_products_prior size is : ", order_products_prior.shape)
orders.head(15)
orders_all= orders[(orders.eval_set=='prior') | (orders.eval_set=='train')]
order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)
#grouped = order_products_all.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'})

#grouped = grouped.sort_values(by='Total_reorders', ascending=False)
#grouped.head(15)
grouped = pd.merge(orders_all, order_products_all[['order_id', 'product_id']], how='left', on=['order_id'])

#grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])
grouped