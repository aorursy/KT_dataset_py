# load dependencies

import pandas as pd # dataframes

import numpy as np # algebra & calculus

import nltk # text preprocessing & manipulation

import matplotlib.pyplot as plt # plotting

import seaborn as sns # plotting



from functools import partial # to reduce df memory consumption by applying to_numeric



color = sns.color_palette() # adjusting plotting style

import warnings

warnings.filterwarnings('ignore') # silence annoying warnings



%matplotlib inline
# import aisles data

aisles = pd.read_csv('../input/aisles.csv', engine='c')

print('Total aisles: {}'.format(aisles.shape[0]))

aisles.head()
# import department data

departments = pd.read_csv('../input/departments.csv', engine='c')

print('Total departments: {}'.format(departments.shape[0]))

departments.head()
# import products data

products = pd.read_csv('../input/products.csv', engine='c')

print('Total products: {}'.format(products.shape[0]))

products.head(5)
# test dataset (submission)

test = pd.read_csv('../input/sample_submission.csv', engine='c')

test.head()
# prior dataset

op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 

                       dtype={'order_id': np.int32, 'product_id': np.int32, 

                              'add_to_cart_order': np.int16, 'reordered': np.int8})



op_prior.head(5)
# orders

orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 

                                                           'user_id': np.int32, 

                                                           'order_number': np.int32, 

                                                           'order_dow': np.int8, 

                                                           'order_hour_of_day': np.int8, 

                                                           'days_since_prior_order': np.float16})



orders.head(5)