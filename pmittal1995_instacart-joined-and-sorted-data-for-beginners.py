# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importing 'orders.csv' and sorting it by 'user_id' and 'order_number'
orders=pd.read_csv('../input/orders.csv')
orders=orders[['user_id','order_number', 'order_id', 'eval_set', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order']].sort_values(['user_id','order_number'])
#first order of user_id==1 can be seen as below:
print(orders.head())
#Examining the different SETs in the data.
print(orders['eval_set'].unique())
#We can drop the 'test' SET
orders.drop(orders[orders['eval_set']=='test'].index,inplace=True)
print(orders['eval_set'].unique())
#We need to map all the orders based on 'order_id' and 'eval_set'. But before we need to import both SETs of 'order_products_*.csv'
order_prior=pd.read_csv('../input/order_products__prior.csv')
print(order_prior.head())
order_train=pd.read_csv('../input/order_products__train.csv')
print(order_train.head())
p_orders=orders[orders['eval_set']=='prior'].drop('eval_set',axis=1)
df_prior=pd.merge(p_orders,order_prior,left_on='order_id',right_on='order_id')
df_prior['eval_set']='prior'
df_prior.head()
t_orders=orders[orders['eval_set']=='train'].drop('eval_set',axis=1)
df_train=pd.merge(t_orders,order_train,left_on='order_id',right_on='order_id')
df_train['eval_set']='test'
df_train.head()
df=pd.concat([df_prior,df_train],ignore_index=True)
df=df.sort_values(['user_id','order_number','add_to_cart_order'])
df.head()
#deleting the dataframes that arent needed
del orders
del order_prior
del order_train
del p_orders
del t_orders
del df_prior
del df_train
#importing 'products.csv', 'aisles.csv' and 'department.csv'
products=pd.read_csv('../input/products.csv')
aisles=pd.read_csv('../input/aisles.csv')
departments=pd.read_csv('../input/departments.csv')
#Merging products and aisles
products_aisles_df= pd.merge(products,aisles,left_on='aisle_id',right_on='aisle_id').sort_values('product_id')
print(products_aisles_df.head())
#Merging products_aisles_df with departments to get 'products_df'
products_df=pd.merge(products_aisles_df,departments,left_on='department_id',right_on='department_id')
print(products_df.head())
#deleting the dataframes not needed
del products
del aisles
del departments
del products_aisles_df
final_df=pd.merge(df,products_df,left_on='product_id',right_on='product_id')
final_df=final_df.sort_values(['user_id','order_number','add_to_cart_order'])
final_df.head(20)
final_df.to_hdf('final.hdf','final_df',mode='w',Table=True)

