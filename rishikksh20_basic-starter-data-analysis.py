# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt # plotting

import seaborn as sns # plotting



from functools import partial # to reduce df memory consumption by applying to_numeric



color = sns.color_palette() # adjusting plotting style

import warnings

warnings.filterwarnings('ignore') # silence annoying warnings



%matplotlib inline

from subprocess import check_output



# Any results you write to the current directory are saved as output.
df_aisles = pd.read_csv('../input/aisles.csv')

print('Total number aisles: {}'.format(df_aisles.shape[0]))

df_aisles.head(5)
df_department = pd.read_csv('../input/departments.csv')

print('Total numbers of depatment: {}'.format(df_department.shape[0]))

df_department.head()
df_prior = pd.read_csv('../input/order_products__prior.csv')

print('Total no. of rows: {}'.format(df_prior.shape[0]))

df_prior.head()
df_orders = pd.read_csv('../input/orders.csv')

print('Total no. of orders: {}'.format(df_orders.shape[0]))

df_orders.head()
df_products = pd.read_csv('../input/products.csv')

print('Total no. of products: {}'.format(df_products.shape[0]))

df_products.head()
df_train = pd.read_csv('../input/order_products__train.csv')

print('Total no. of rows: {}'.format(df_train.shape[0]))

df_train.head()
df_complete_prod = pd.merge(left=pd.merge(left=df_products, right=df_department, how='left'), right=df_aisles, how='left').drop(['department_id', 'aisle_id'], axis=1)

df_complete_prod.head()
plt.figure(figsize=(13, 5))

df_complete_prod.groupby(['department']).count()['product_id'].copy().sort_values(ascending=False).plot(kind='bar', title='# of Product per Department')
plt.figure(figsize=(13, 5))

df_complete_prod.groupby(['aisle']).count()['product_id'].copy().sort_values(ascending=False)[:15].plot(kind='bar', title='# of Product per Aisle')
df_prior_product = pd.merge(left=df_prior, right=df_products, how='left').drop('product_id', axis=1)

df_prior_product.head()
df_prior_prod_dept = pd.merge(left=df_prior, right=df_complete_prod, how='left').drop('product_id', axis=1)

df_prior_prod_dept.head()
plt.figure(figsize=(13, 5))

df_prior_product.groupby(['product_name']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Products')
plt.figure(figsize=(13, 5))

df_prior_prod_dept.groupby(['department']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Departments')
plt.figure(figsize=(13, 5))

df_prior_prod_dept[df_prior_prod_dept.reordered>0].groupby(['product_name']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Reordered Products')
print("Number of products reordered are : {}".format(len(df_prior_prod_dept[df_prior_prod_dept.reordered>0].product_name.value_counts().unique())))
plt.figure(figsize=(13, 5))

df_prior_product.groupby(['reordered']).count()['order_id'].copy().sort_values(ascending=False).plot(kind='bar', title='Reordered ratio')
df_prior_product.reordered.value_counts()
print("Percentage of Reordering in Prior Dataset is : {} %".format(df_prior_product.reordered.sum() / df_prior_product.shape[0]*100))
df_order_reordered = df_prior_product.groupby("order_id")["reordered"].aggregate("sum").reset_index()

df_order_reordered["reordered"].ix[df_order_reordered["reordered"]>1] = 1

df_order_reordered.reordered.value_counts()
print("{} % of the orders in the Prior set has no-reordered products and the rest of them has atleast one reordered products.".format((1-df_order_reordered.reordered.sum() / df_order_reordered.shape[0])*100))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt # plotting

import seaborn as sns # plotting



from functools import partial # to reduce df memory consumption by applying to_numeric



color = sns.color_palette() # adjusting plotting style

import warnings

warnings.filterwarnings('ignore') # silence annoying warnings



%matplotlib inline

from subprocess import check_output



# Any results you write to the current directory are saved as output.
df_aisles = pd.read_csv('../input/aisles.csv')

print('Total number aisles: {}'.format(df_aisles.shape[0]))

df_aisles.head(5)
df_department = pd.read_csv('../input/departments.csv')

print('Total numbers of depatment: {}'.format(df_department.shape[0]))

df_department.head()
df_prior = pd.read_csv('../input/order_products__prior.csv')

print('Total no. of rows: {}'.format(df_prior.shape[0]))

df_prior.head()
df_orders = pd.read_csv('../input/orders.csv')

print('Total no. of orders: {}'.format(df_orders.shape[0]))

df_orders.head()
df_products = pd.read_csv('../input/products.csv')

print('Total no. of products: {}'.format(df_products.shape[0]))

df_products.head()
df_train = pd.read_csv('../input/order_products__train.csv')

print('Total no. of rows: {}'.format(df_train.shape[0]))

df_train.head()
df_complete_prod = pd.merge(left=pd.merge(left=df_products, right=df_department, how='left'), right=df_aisles, how='left').drop(['department_id', 'aisle_id'], axis=1)

df_complete_prod.head()
plt.figure(figsize=(13, 5))

df_complete_prod.groupby(['department']).count()['product_id'].copy().sort_values(ascending=False).plot(kind='bar', title='# of Product per Department')
plt.figure(figsize=(13, 5))

df_complete_prod.groupby(['aisle']).count()['product_id'].copy().sort_values(ascending=False)[:15].plot(kind='bar', title='# of Product per Aisle')
df_prior_product = pd.merge(left=df_prior, right=df_products, how='left').drop('product_id', axis=1)

df_prior_product.head()
df_prior_prod_dept = pd.merge(left=df_prior, right=df_complete_prod, how='left').drop('product_id', axis=1)

df_prior_prod_dept.head()
plt.figure(figsize=(13, 5))

df_prior_product.groupby(['product_name']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Products')
plt.figure(figsize=(13, 5))

df_prior_prod_dept.groupby(['department']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Departments')
plt.figure(figsize=(13, 5))

df_prior_prod_dept[df_prior_prod_dept.reordered>0].groupby(['product_name']).count()['order_id'].copy().sort_values(ascending=False)[:10].plot(kind='bar', title='Top Reordered Products')
print("Number of products reordered are : {}".format(len(df_prior_prod_dept[df_prior_prod_dept.reordered>0].product_name.value_counts().unique())))
plt.figure(figsize=(13, 5))

df_prior_product.groupby(['reordered']).count()['order_id'].copy().sort_values(ascending=False).plot(kind='bar', title='Reordered ratio')
df_prior_product.reordered.value_counts()
print("Percentage of Reordering in Prior Dataset is : {} %".format(df_prior_product.reordered.sum() / df_prior_product.shape[0]*100))
df_order_reordered = df_prior_product.groupby("order_id")["reordered"].aggregate("sum").reset_index()

df_order_reordered["reordered"].ix[df_order_reordered["reordered"]>1] = 1

df_order_reordered.reordered.value_counts()
print("{} % of the orders in the Prior set has no-reordered products and the rest of them has atleast one reordered products.".format((1-df_order_reordered.reordered.sum() / df_order_reordered.shape[0])*100))