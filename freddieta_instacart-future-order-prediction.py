import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import re
import seaborn as sns
color = sns.color_palette()
from tqdm import tqdm # progress bar

# Limit floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '%.3f' % x)

plt.style.use('fivethirtyeight')
%matplotlib inline 

# Increase default figure and font sizes for easier viewing.
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

from sklearn.model_selection import train_test_split
# List of files
from subprocess import check_output
print(check_output(["ls", "../input/instacart-market-basket-analysis/"]).decode("utf8"))
order_products_train = pd.read_csv("../input/instacart-market-basket-analysis/order_products__train.csv")
order_products_prior = pd.read_csv("../input/instacart-market-basket-analysis/order_products__prior.csv")
orders = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")
products = pd.read_csv("../input/instacart-market-basket-analysis/products.csv")
aisles = pd.read_csv("../input/instacart-market-basket-analysis/aisles.csv")
departments = pd.read_csv("../input/instacart-market-basket-analysis/departments.csv")
order_products_train.head() # order_id and product_id

# Includes training orders
# Indicates whether product is a reorder (one or zero) via reordered variable
order_products_prior.head() # order_id and product_id

# Like above, indicates whether a product is a reorder
orders.head() # order_id

# Includes all orders (prior, train, test)
products.head() # product_id
aisles.head() # aisle_id
departments.head() # department_id
# Check for missing data
orders.isnull().sum()
print(orders.shape, order_products_prior.shape, order_products_train.shape, aisles.shape, products.shape, departments.shape)
orders.columns
combine_dataset = orders.groupby('eval_set')['order_id'].aggregate({'Total_orders': 'count'}).reset_index()

combine_dataset
combine_dataset  = combine_dataset.groupby(['eval_set']).sum()['Total_orders'].sort_values(ascending=False)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(10,10))
sns.barplot(combine_dataset.index, combine_dataset.values, palette="RdBu")
plt.ylabel('Number of Orders', fontsize=14)
plt.title('Types of Datasets', fontsize=16)
plt.show()
# Convert variables into categories:

aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')
# Create DataFrame with ORDERS and PRODUCTS purchased on PRIOR ORDERS
# po = 'orders' + 'order_products_prior' (only prior orders)

# For every 'user_id', there is 'order_id' and 'product_id'

po = orders.merge(order_products_prior, on='order_id', how='inner') #merge fx to help return info with matching values in both df.
po.head()

po.head()
# Create a customer/user dataframe
# ID highest order # in each of these groups
# Save column into a dataframe

user = po.groupby('user_id')['order_number'].max().to_frame('c_total_orders')

user.head(2)
# prob reordered = 
## (total times of reorder) / (total # of ordered products)

# (total times of reorder) : all the times customer has reordered
# (total # of ordered products) : products that have been purchased; binary reorder value (0,1)

# 'reordered' info from 'order_products_prior'('po')
# Calculate mean of reordered

c_reorder = po.groupby('user_id')['reordered'].mean().to_frame('c_reordered_ratio')

# c_reorder = c_reorder.reset_index()

c_reorder.head()

dow = po.groupby('user_id')['order_dow'].mean().to_frame('average_dow')

dow.head()
# Left join to keep users/customers created to be in the user DataFrame:

user = user.merge(c_reorder, on='user_id', how='left')

user.head()

user = user.merge(dow, on='user_id', how='left')

user.head()
po.head(0)
# Create product dataframe to store results
# Total number of purchases (count)

prod = po.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')

prod = prod.reset_index() # reset to bring 'product_id' from index to column

prod.head()
# Remove products < 50 purchases
# Create groups for each product and keep groups with more than 50 rows.

p_reorder = po.groupby('product_id').filter(lambda x: x.shape[0] > 50)

p_reorder.head()

p_reorder.columns
# Group products

# Calculate mean of reorders to get reorder ratio
## (# times product reordered)/ (total # times has been ordered)
## reordered = 1, not reordered = 0

p_reorder = p_reorder.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')

p_reorder = p_reorder.reset_index()

p_reorder.head()

# Average order of product added to cart

addtocart = po.groupby('product_id')['add_to_cart_order'].mean().to_frame('Ave_Added_To_Cart')

addtocart = addtocart.reset_index()

addtocart.head()
# Combine 'prod' and 'reorder' dataframes together:

prod = prod.merge(p_reorder, on='product_id', how='left')

prod.head()
# Merge in 'addtocart' columns

prod = prod.merge(addtocart, on='product_id', how='left')

prod.head()
# Replace NaN values in 'p_reorder_ratio' column:

prod['p_reorder_ratio'] = prod['p_reorder_ratio'].fillna(value=0)

prod.head()
po.columns
# Create unique groups for each combo of user and product.
# Get how many times each user bought a product using .count()
# New dataframe 'userprod'

userprod = po.groupby(['user_id','product_id'])['order_id'].count().to_frame('userprod_total_bought')

userprod = userprod.reset_index() 

userprod.head()
"""
reorder_ratio = times_bought / order_range

This ratio will help us see how many times a user bought a product out of how many times they had 
a chance to purchase it starting from the first purchase of the item.

"""

## times_bought : number of times user bought a product
## order_range : total orders placed since user's order of product

# order_range = 
## total_orders : total number of orders per user
## first_order_num : order number where user bought product for first time

"""
final ratio is our 'userprod_reorder_ratio'

"""
# times_bought

# Group 'user_id' and 'product_id'
# Count events of 'order_id' per group

times_bought = po.groupby(['user_id', 'product_id'])[['order_id']].count()

times_bought.columns = ['times_bought']

times_bought.head()

po.columns
"""total_orders"""
# Calculate total orders of each user

total_orders = po.groupby('user_id')['order_number'].max().to_frame('total_orders')

total_orders.head()

"""first_order_num"""
# Calculate first order number for every user and product bought
## Group 'user_id' and 'product_id' 
## Select 'order_number' column and get .min value

first_order_num = po.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_num')
first_order_num = first_order_num.reset_index()
first_order_num.head()
# Merge 'total_orders' and 'first_order_num' dataframes

# Join right for'first_order_num' because it refers to unique combinations of user/prod
# 'total_orders' apply to all users

order_range = pd.merge(total_orders, first_order_num, on='user_id', how='right')
order_range.head()

"""order_range"""
# Within 'pre_order_range', subtract 'first_order_num' from 'total_orders'
# Add 1 for the difference between the first order where the product has been purchased.

order_range['order_range'] = order_range.total_orders - order_range.first_order_num + 1

order_range.head()

# reorder_ratio = times_bought / order_range

## Both variables from combination of users/products; any join will do.
## Merge 'times_bought' and 'order_range'

reorder_ratio = pd.merge(times_bought, order_range, on=['user_id','product_id'], how='left')

reorder_ratio.head()
# Calculate reorder_ratio = (times_bought / order_range) --> 'userprod_reorder_ratio'
# Add column for 'userprod_reorder_ratio'

reorder_ratio['userprod_reorder_ratio'] = reorder_ratio.times_bought / reorder_ratio.order_range

reorder_ratio.head()
# Only need columns 'user_id', 'product_id', and 'userprod_reorder_ratio'
# Remove other columns

reorder_ratio = reorder_ratio.drop(['times_bought', 'total_orders', 'first_order_num', 
                                   'order_range'], axis=1)

reorder_ratio.head()

userprod.head()
# Merge 'reorder_ratio' with 'userprod'
# Left join to keep all user/products made in 'userprod'

userprod = userprod.merge(reorder_ratio, on=['user_id','product_id'], how= 'left')

userprod.head()

po[po.user_id==1].head()
# Create variable that keeps 'order_number' in reverse order

## This will indicate last order as a sequence (1st, 2nd, etc) from the end.
## Need max order number for 'user_id' and subtract 'order_number' from it.


"""
order_number_rev = max order number - order_number + 1 
"""
# .transform(max) to request highest number of 'order_number' column for each group
# Subtract 'order_number' from each row with '- po.order_number'
# Add +1 because it's the last order to be marked first

po['order_number_rev'] = po.groupby('user_id')['order_number'].transform(max) - po.order_number + 1 



po.head(15)

# Confirm it's been applied to users other than the first

po[po.user_id==4].head(10)
# Keep last few orders of each user with 'order_number_rev':

po3 = po[po.order_number_rev <= 3]

po3.head(15)
# Group users and products to see how frequent a customer ordered on their last three orders 'last_three_orders_times':

""" last_three = (times user bought product on its last 3 orders) / (total orders) """

last_three = po3.groupby(['user_id','product_id'])[['order_id']].count()

last_three.columns = ['last_three_orders_times']

last_three.head()
# Merge 'last_three' dataframe to the 'userprod' dataframe:
# Left join to keep all user-products on 'userprod' dataframe

userprod = userprod.merge(last_three, on=['user_id','product_id'], how='left')

userprod.head()
# Fill in NaN values (see product_id 10326 with NaN)

userprod = userprod.fillna(0)
userprod.head(3)
# Merge 'user' with 'userprod' dataframe, store into new dataframe 'data'
# Match 'user_id' key
# Left join to keep all data from 'userprod'

data = userprod.merge(user, on='user_id', how='left')

data.head()


# Merge 'prod' with 'data' dataframe
# Match 'product_id' key
# Left join to keep data from 'data' (features of users and combination of users/products)

data = data.merge(prod, on='product_id', how='left')

data.head()
orders.columns
data.columns
# From 'orders' dataframe, select 'eval_set', 'order_id', and 'user_id' (matching key) and 
# merge into 'data' dataframe:

## 'eval_set' : train/test type
## 'order_id' : future orders
## 'user_id' : will be the matching key during merge

""" 'data_train' will contain 'eval_set', 'order_id', and 'user_id' """

future_orders = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]

future_orders = future_orders[ ['user_id', 'eval_set', 'order_id'] ]

future_orders.head(10)


# Transfer info of 'future_orders' to 'data' dataframe:

data = data.merge(future_orders, on='user_id', how='left')

data.head()
order_products_train.columns
# Keep users labeled 'train'

data_train = data[data.eval_set=='train']

# Create variable that will show all products that the users buy in their future order
## Source: 'order_products_train'
## Matching keys of 'product_id' and 'order_id'
## Left join to 'data_train' to keep all observations

data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )

data_train.head()
# Remove NaN in column 'reordered' and set to zero
## reordered status (1 or zero)

data_train['reordered'] = data_train['reordered'].fillna(0)

# data_train.head()
data_train.reordered.isnull().sum()
# 'user_id' and 'product_id' as index

# data_train = data_train.set_index(['user_id','product_id'])

# Remove non-predictor columns

data_train = data_train.drop(['eval_set', 'order_id'], axis=1)

data_train.head()
# Will be used for prediction model
# Keep users who have eval_set as 'test'

data_test = data[data.eval_set=='test']

data_test.head()
# Set 'user_id' and 'pruduct_id' as index to describe each row

data_test = data_test.set_index(['user_id', 'product_id'])

# Remove non-predictor variables; 'eval_set', 'order_id'

data_test = data_test.drop(['eval_set','order_id'], axis=1)

data_test.head()
# Check where XGBoost was installed (pip3 install xgboost)
# Append that directory to sys.path
# Finally import xgboost

import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
# Import the xgboost package
import xgboost as xgb
data_train.head(0), data_test.head(0)
# Split dataframe to: 'X_train' and 'y_train' ; axis= 1

X_train, y_train = data_train.drop('reordered', axis = 1), data_train.reordered
# XGBoost parameters: 'eval_metric', 'max_depth', 'colsample_bytreeι', 'subsample'

parameters = {'eval_metric':'logloss', 
              'max_depth':'5', 
              'colsample_bytree':'0.4', # 0.3 - 0.8 if many columns
              'subsample':'0.8',
              'n_estimators':100, # 100 if large data, 1000 if med-low
              'verbose': 1 # prints progress - takes awhile to fit 'model'
             }
# Instantiate XGBClassifier() 'xgbc'

xgbc = xgb.XGBClassifier(objective='binary:logistic', parameters=parameters, num_boost_round=10)
# Train the model using xgbc.fit on train data

model = xgbc.fit(X_train, y_train)
# Plot model to observe feature importance:
xgb.plot_importance(model)

from sklearn.model_selection import GridSearchCV

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# 
# TRIM 'data_train' to 10%

# data_train.reset_index(inplace=True)

trimmed_data_train = data_train.loc[data_train.user_id.isin(data_train.user_id.drop_duplicates().sample(frac=0.1, random_state=25))].set_index(['user_id', 'product_id'])

X_train, y_train = trimmed_data_train.drop('reordered', axis=1), trimmed_data_train.reordered
# Adjust Booster's parameters (range)

gridparam = {"max_depth":[5,10], 
             "colsample_bytree":[0.3, 0.4]}  
# Instantiate XGBClassifier()

# xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', num_boost_round=10)

# n_jobs=-1

xgbc = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', num_boost_round=10)

# Define how to train different models:
## xgbc
## gridparam
# .GridSearchCV() : to tune parameters to find best accuracy.

# gridsearch = GridSearchCV(xgbc, gridparam, cv=3, verbose=2, n_jobs=-1)
gridsearch = GridSearchCV(xgbc, gridparam, cv=3, verbose=2)

# n_jobs -1 number of jobs to run in parallel. -1 runs all.
# Memory leak: https://stackoverflow.com/questions/55848101/memory-leak-using-gridsearchcv
## Putting 'n_jobs=-1' in classifier() instead of gridsearch
"""
# Train models with combination of parameters.
# GridSearch function : to tune parameters to find best accuracy.

model_best = gridsearch.fit(X_train, y_train)

# print("Top parameters are: /n", gridsearch.best_parameters_)

"""
import scipy
scipy.test()
model_best = gridsearch.fit(X_train, y_train)
# With test data predict values with 'model'

prediction_test = model.predict(data_test) #.astype(int)
prediction_test[0:20] # Display the first 10 predictions of numpy array
# Save prediction

data_