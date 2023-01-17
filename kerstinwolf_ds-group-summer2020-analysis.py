# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import bqplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import sklearn

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
order_products_train_df = pd.read_csv("../input/instacart-market-basket-analysis/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/instacart-market-basket-analysis/order_products__prior.csv")
orders_df = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")
products_df = pd.read_csv("../input/instacart-market-basket-analysis/products.csv")
aisles_df = pd.read_csv("../input/instacart-market-basket-analysis/aisles.csv")
departments_df = pd.read_csv("../input/instacart-market-basket-analysis/departments.csv")
print("Setup Complete")
order_products_train_df.head()
order_products_prior_df.head()
orders_df.head()
products_df.head()
aisles_df.head()
departments_df.head()
# merging csv files
# what day of the week is 1 in order_dow? Sunday or Monday?
# reordered appears to work in binary 1s and 0s. 1 means yes?
# add_to_cart_order refers to the order in which the product was added to the cart.
# the amount of each product added to each order/cart is not provided.

order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, orders_df, on='order_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')
order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')
order_products_prior_df.head()
order_products_prior_df['product_id'] == 33120
# sorts out product with the product id of 33120
# loc gets rows with particular labels in it
egg_whites_order_info = order_products_prior_df.loc[order_products_prior_df['product_id'] == 33120, :]
# All egg whites purchased on a Sunday or Monday (what the first day of the week is).
    # While not super useful now, it might be something to keep in mind. Also, this is currently not sorted in any way.
# Data not given: is the week beginning on Sunday or Monday for order_dow? 
# Interesting idea: making a chart that sorts orders by order_dow.
egg_whites_order_info.loc[egg_whites_order_info['order_dow'] == 1, :]
# orders the egg whites orders by ascending value of user id (user id 155 likes the egg whites)
sorted_egg_whites = egg_whites_order_info.sort_values('user_id', ascending = True)
sorted_egg_whites
# gives the first 15 rows of the previous chart. While the data in this chart isn't the most useful
    # it is useful in that I definitely will need to use iloc later. So this is more for myself than anything.
# iloc gets index position
# What is the difference between .head() and .iloc[]? So far, they appear to be interchangeable.
# Correction from earlier block: User 155 loves egg whites. (total of 14 separate orders that began on their 2nd order.)
sorted_egg_whites.iloc[0:15]
# This is how many times a specific product is on an order
# well, this looks weird, but despite the count being in every column, the count is correct.
# is there a nicer way of coding this that's less messy? Yes, see next block.
# also, there are 49,677 unique product names in this dataset.
product_count = order_products_prior_df.groupby('product_name').count()
product_count
# quick test to check count
mech_pencil_orders = order_products_prior_df.loc[order_products_prior_df['product_name'] == '#2 Mechanical Pencils', :]
mech_pencil_orders.product_name.count()
# Looking for null values
# I'm beginning to think that axis = 0 refers to columns and axis = 1 means rows
# This may mean that days_since_prior_order may have null values

null_value_columns = order_products_prior_df.isnull().any(axis = 0)
print(null_value_columns)
# list of all rows containing a null value. In this case, the nulls appear to be in days_since_prior_order.
# This makes sense since a new customer wouldn't have an order prior to their first.
# This means there are likely over 2 million new customers in this dataset. See row count of 2,078,068 at bottom.
null_value_rows = order_products_prior_df.isnull().any(axis = 1)
order_products_prior_df[null_value_rows]
# shows unique product names
# semi-failed attempt at data cleaning
# this was an attempt to make sure everything is spelled correctly, but that didn't work since I can't see all the names in the list in Kaggle.
unique_product_names = order_products_prior_df['product_name'].unique()
unique_product_names
# getting a single row for each order_id in prior set, not training set
# useful for seeing order_id with the order_dow, order_hour_of_day, and days_since_prior_order

order_id_groups = order_products_prior_df.groupby('order_id').first()
order_id_sorted_groups = order_id_groups.sort_values('order_id', ascending = True)
order_id_sorted_groups
# Well, this way was easier than the prior block. This includes training data, though.
# I organized it to makes sure I wasn't getting duplicates of order ids
orders_df.sort_values('order_id', ascending = True)
plt.figure(figsize = (15,8))
sns.countplot(x = 'days_since_prior_order', data = orders_df)
plt.ylabel('Count')
plt.xlabel('Days since prior order')
plt.title('Frequency of days since prior order')
plt.figure(figsize = (12,8))
sns.countplot(x = "order_hour_of_day", data = orders_df)
plt.ylabel('Count')
plt.xlabel('Hour of day')
plt.title("Frequency of orders by hour of day")
# Info learned: days of the week start from 0 and go to 6. Does not start on 1 like originally thought.
# Still unknown if 0 represents Sunday or Monday
# counts each individual order including orders from the training data as well as prior data.

plt.figure(figsize = (12,8))
sns.countplot(x = 'order_dow', data = orders_df)
plt.ylabel('Count')
plt.xlabel('Day of the week')
plt.title('Frequency of orders each day of the week')