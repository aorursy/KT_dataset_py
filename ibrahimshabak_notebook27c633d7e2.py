# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

%matplotlib inline 
DATA_FOLDER = '/kaggle/input/competitive-data-science-final-project/'



transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))

items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))

item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))

shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
print(transactions.shape)

transactions.head()
print(items.shape)

items.head()
print(item_categories.shape)

item_categories.head()
print(shops.shape)

shops.head()
transactions['date'] = pd.to_datetime(transactions['date'], format='%d.%m.%Y')

trans = transactions[(transactions['date'].dt.year == 2014) & (transactions['date'].dt.month == 9)]

# VARIABLE

max_revenue = (trans['item_price'] * trans['item_cnt_day']).groupby(trans['shop_id']).sum().max()
trans = transactions[(transactions['date'].dt.year == 2014) & (transactions['date'].dt.month >= 6) & (transactions['date'].dt.month <= 8)]

trans = pd.merge(trans, items, how='left', on='item_id')

# PUT YOUR ANSWER IN THIS VARIABLE

category_id_with_max_revenue = (trans['item_price'] * trans['item_cnt_day']).groupby(trans['item_category_id']).sum().idxmax()
trans = transactions.groupby('item_id')['item_price']

# PUT YOUR ANSWER IN THIS VARIABLE

num_items_constant_price = (trans.nunique() == 1).sum()
shop_id = 25

trans = transactions[(transactions['shop_id'] == shop_id) & (transactions['date'].dt.year == 2014) & (transactions['date'].dt.month == 12)]

trans = trans.groupby('date')['item_cnt_day'].sum()



total_num_items_sold = trans.values # YOUR CODE GOES HERE

days = trans.index # YOUR CODE GOES HERE



# Plot it

plt.plot(days, total_num_items_sold)

plt.ylabel('Num items')

plt.xlabel('Day')

plt.title("Daily revenue for shop_id = 25")

plt.show()



total_num_items_sold_var = trans.var() # PUT YOUR ANSWER IN THIS VARIABLE
