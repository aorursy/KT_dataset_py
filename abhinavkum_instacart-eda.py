# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib notebook

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import pandas_profiling as pp

import cufflinks as cf

import plotly.offline

cf.go_offline()



cf.set_config_file(offline = False, world_readable = True)
orders_df = pd.read_csv("../input/orders.csv")

orders_df.head(3)

pp.ProfileReport(orders_df.iloc[:,[1,2,4,5,6]])  # considering eval_set, order_dow, orer_hour_of_day and days_since_prior_order
# 6% unique userID

no_of_user = len(np.unique(orders_df['user_id']))

print('NO. of users are', no_of_user )

print('{:.1f}%  as stated in Pandas profiling'.format( no_of_user / orders_df.shape[0] *100))
# checking userid in different eval_set

def checklen(x):

    return len(np.unique(x))

eval_users = orders_df.groupby(['eval_set']).agg({'user_id': checklen})

eval_users['Percentage of Total User'] = eval_users.apply(lambda x: round(x / no_of_user*100,1))

eval_users
unique_user_test = np.unique(orders_df.query('eval_set == "test"')['user_id'])

unique_user_train = np.unique(orders_df.query('eval_set == "train"')['user_id'])

i = 0

len([user for user in unique_user_test if user in unique_user_train])
# how many time a user order?



orders_df.groupby('user_id').agg({'order_number' :  'max'})['order_number'].value_counts().iplot(kind = 'bar',

                                                                                                 title = 'Maximum order Number')

orders_df.groupby(['order_dow', 'order_hour_of_day']).agg({'order_number' : 'count'}).iplot()
grouped_df = orders_df.groupby(['order_dow', 'order_hour_of_day']).agg({'order_number' : 'count'}).reset_index()

grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

grouped_df.iplot(kind='heatmap',colorscale='-rdbu', title = 'No. of order on various Time of day and days of week',

                 xTitle = 'Order Day of Week', yTitle = 'Order Time of day')
orders_df.days_since_prior_order.value_counts().iplot(kind = 'bar', title = 'Frequency distribution by days since prior order',

                                                      xTitle = 'Days since prior order', yTitle = 'Count')
# Now lets check another file, order_products __prior. means it have previous order of all users.

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

order_products_prior_df.head(3)
pp.ProfileReport(order_products_prior_df.iloc[:, 2:]) # Creating Profile Report of add_to_Cart_order and reordered
order_products_prior_df.groupby('order_id').agg({'add_to_cart_order' : 'max'})['add_to_cart_order'].value_counts().iplot(

                                                                                    kind = 'bar',

                                                                                    xTitle = 'Maximum no. of product in order',

                                                                                    yTitle = 'Count')
def percent(x):

    return sum(x)/len(x)*100

order_products_prior_df.groupby('order_id').agg({'reordered':percent})['reordered'].value_counts(normalize  = True).iloc[:5].iplot(kind = 'barh', 

                                                                             title = 'What Percentage of order is Reorder',

                                                                             xTitle = 'Percentage of Total order',

                                                                             yTitle = 'Percentage of reorder in paticular order %')
# Now lets check another file, order_products __train. means it have recent order of all 61% users .

order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_train_df.head(3)
pp.ProfileReport(order_products_train_df.iloc[:, 2:]) # Creating Profile Report of add_to_Cart_order and reordered
order_products_train_df.groupby('order_id').agg({'add_to_cart_order' : 'max'})['add_to_cart_order'].value_counts().iplot(

                                                                                    kind = 'bar',

                                                                                    xTitle = 'Maximum no. of product in order',

                                                                                    yTitle = 'Count')
order_products_train_df.groupby('order_id').agg({'reordered': percent})['reordered'].value_counts(normalize  = True).iloc[:5].iplot(kind = 'bar', 

                                                                             title = 'What Percentage of order is Reorder',

                                                                             xTitle = 'Percentage of reorder in paticular order %',

                                                                             yTitle = 'Percentage of Total order')
# Now lets check another file, products. As name suggest it have detail of product.

products_df = pd.read_csv("../input/products.csv")

products_df.head(3)
aisles_df = pd.read_csv("../input/aisles.csv")

aisles_df.head(3)
departments_df = pd.read_csv("../input/departments.csv")

departments_df.head(3)
# merging all three

products_df = pd.merge(left = products_df, right = aisles_df, how = 'left', on = 'aisle_id') 

products_df = pd.merge(left = products_df, right = departments_df, how = 'left', on = 'department_id')

products_df.head(3)
del departments_df, aisles_df
# Number of Products

products_df.shape[0]
pp.ProfileReport(products_df.iloc[:,[1,4,5]])
products_df.department.value_counts().reset_index().iplot(kind = 'pie', labels = 'index', values = 'department')
(products_df.aisle.value_counts(normalize = True, sort = False)*100).iplot(kind = 'bar')


products_df.groupby('department').agg({'aisle':checklen}).sort_values(by = 'aisle').iplot(kind = 'bar', title = 'No. of Aisles in Department',

                                                               xTitle = 'Department', yTitle = 'Count')
# Now joining Prior with products detail

order_products_prior_df = pd.merge(left = order_products_prior_df, right = products_df, how = 'left', on = 'product_id')

del products_df

order_products_prior_df.head(3)

department_wise_product_add = order_products_prior_df.groupby(['add_to_cart_order', 'department']).agg({'product_name' : 'count'}).reset_index()

department_wise_product_add = department_wise_product_add.pivot('department', 'add_to_cart_order', 'product_name')

department_wise_product_add = department_wise_product_add /department_wise_product_add.sum() * 100

department_wise_product_add.sort_values(by = [1,2], axis = 0, ascending = False, inplace = True)

department_wise_product_add.iloc[:,:5].iplot(title = 'How many time Department\'s product added in specified cart orde',

                                             yTitle = 'percentage',

                                             xTitle = 'Department')
department_wise_product_add = order_products_prior_df.groupby(['reordered', 'department']).agg({'product_name' : 'count'}).reset_index()

department_wise_product_add = department_wise_product_add.pivot('department', 'reordered', 'product_name')

department_wise_product_add = department_wise_product_add /department_wise_product_add.sum() * 100

department_wise_product_add.sort_values(by = [1], axis = 0, ascending = False, inplace = True)

department_wise_product_add.iloc[:,:].iplot(title = 'How many time Department\'s product reordered',

                                             yTitle = 'percentage',

                                             xTitle = 'Department')
aisles_wise_product_add = order_products_prior_df.groupby(['add_to_cart_order', 'aisle']).agg({'product_name' : 'count'}).reset_index()

aisles_wise_product_add = aisles_wise_product_add.pivot('aisle', 'add_to_cart_order', 'product_name')

aisles_wise_product_add.sort_values(by = [1,2,3,4,5], axis = 0, ascending = False, inplace = True)

aisles_wise_product_add = aisles_wise_product_add / aisles_wise_product_add.sum() * 100

aisles_wise_product_add.iloc[:20,:5].iplot(title = 'How many time aisles\'s product added in specified cart order',

                                           xTitle = 'Top 20 aisles',

                                           yTitle = 'Percentage')
aisles_wise_product_add = order_products_prior_df.groupby(['reordered', 'aisle']).agg({'product_name' : 'count'}).reset_index()

aisles_wise_product_add = aisles_wise_product_add.pivot('aisle', 'reordered', 'product_name')

aisles_wise_product_add.sort_values(by = [1], axis = 0, ascending = False, inplace = True)

aisles_wise_product_add = aisles_wise_product_add / aisles_wise_product_add.sum() * 100

aisles_wise_product_add.iloc[:20,:].iplot(title = 'How many time aisles\'s product reordered',

                                           xTitle = 'Top 20 aisles',

                                           yTitle = 'Percentage')
product_wise_product_add = order_products_prior_df.groupby(['add_to_cart_order', 'product_name']).agg({'aisle' : 'count'}).reset_index()

product_wise_product_add = product_wise_product_add.pivot('product_name', 'add_to_cart_order', 'aisle')

product_wise_product_add.sort_values(by = [1,2,3,4,5], axis = 0, ascending = False, inplace = True)

product_wise_product_add = product_wise_product_add / product_wise_product_add.sum() * 100

product_wise_product_add.iloc[:20,:5].iplot(title = 'How many time product added in specified cart order',

                                           xTitle = 'Top 20 product',

                                           yTitle = 'Percentage')
product_wise_product_add = order_products_prior_df.groupby(['reordered', 'product_name']).agg({'aisle' : 'count'}).reset_index()

product_wise_product_add = product_wise_product_add.pivot('product_name', 'reordered', 'aisle')

product_wise_product_add.sort_values(by = [1,0], axis = 0, ascending = False, inplace = True)

product_wise_product_add = product_wise_product_add / product_wise_product_add.sum() * 100

product_wise_product_add.iloc[:20,1].iplot(title = 'How many time product reordered',

                                           xTitle = 'Top 20 product',

                                           yTitle = 'Percentage')
order_products_prior_df = pd.merge(left = order_products_prior_df, right = orders_df, how = 'left', on = 'order_id')

del orders_df

order_products_prior_df.head()
dept_order_hour = order_products_prior_df.groupby(['department', 'order_hour_of_day']).agg({'order_id':'count'}).reset_index()

dept_order_hour = dept_order_hour.pivot('order_hour_of_day', 'department', 'order_id')

dept_order_hour.iplot(kind = 'heatmap', colorscale='Spectral')