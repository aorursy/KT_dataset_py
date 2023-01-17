# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Since training data is huge, so I am planning to read few millions of rows from the csv file.

train = pd.read_csv("../input/train.csv", nrows=25000000, parse_dates=['date'],index_col='id')



#print the last 10 rows of the data, this will help us to think what we can dow with the data.

train.tail(5)
items = pd.read_csv("../input/items.csv")
train_items = pd.merge(train, items, how='inner')

train_items.tail(5)
#Lets find out most popular item ordered by people across the 6 millions rows we have read.

#We will group by item_nbr and add the unit sales.

df = train_items['unit_sales'].groupby(train_items['item_nbr']).sum()

#In order to find top 10 popular items we will sort the numpy array and pick the top 10 from

#the list.

df = df.sort_values()

df_highest = df.nlargest(n=10)



#Plot the highest list of items.

df_highest.plot(kind='bar',figsize = (10,10),  title = "Top 10 items sold across all stores")

plt.show()
#Next we find lowest/less demand product. We use nsmallest to find the bottom 10 items,

#probably it doesn;t matter even if we stock them.

df_lowest = df.nsmallest(n=10)

df_lowest.plot(kind='bar',figsize = (10,10),  title = "Bottom 10 items sold")

plt.show()
#Next we could find out popular items in a given year. This will be useful to find out 

#if there were any new items introduced in the recent times.

#In order to do that we need to covert the date field into python date format and then

# extract various fields from it.



train_items['date'] = pd.to_datetime(train_items['date'], format='%Y-%m-%d')

train_items['day_item_purchased'] = train_items['date'].dt.day

train_items['month_item_purchased'] =train_items['date'].dt.month

train_items['quarter_item_purchased'] = train_items['date'].dt.quarter

train_items['year_item_purchased'] = train_items['date'].dt.year
train_items.drop('date', axis=1, inplace=True)
#Lets print out new training dataset

print (train_items.tail(2))
df_year = train_items.groupby(['quarter_item_purchased', 'item_nbr'])['unit_sales'].sum()

df_year = df_year.sort_values()

df_year_highest = df_year.nlargest(n=10)

#Plot the highest list of items.

df_year_highest.plot(kind='bar',figsize = (10,10),  title = "Top items sold Quarterly")

plt.show()
plt.figure(figsize=(9,10))

df_items = train_items.groupby(['family'])['unit_sales'].sum()

df_items = df_items.sort_values()

df_items_highest = df_items.nlargest(n=10)

plt.pie(df_items_highest, labels=df_items_highest.index,shadow=False,autopct='%1.1f%%')

plt.tight_layout()

plt.show()

grocery_info = train_items.loc[train_items['family'] == 'GROCERY I']
plt.figure(figsize=(12,12))

#print (grocery_info.tail(2))

plt.plot(grocery_info['day_item_purchased'],grocery_info['unit_sales'])

plt.show()
plt.figure(figsize=(9,10))

df_items = train_items.groupby(['family','perishable'])['unit_sales'].sum()

df_items = df_items.sort_values()

df_items_perish_highest = df_items.nlargest(n=10)

plt.pie(df_items_perish_highest, labels=df_items_perish_highest.index,shadow=False,autopct='%1.1f%%')

plt.tight_layout()

plt.show()
transaction = pd.read_csv("../input/transactions.csv")
transaction['date'] = pd.to_datetime(transaction['date'], format='%Y-%m-%d')

transaction['day_item_purchased'] = transaction['date'].dt.day

transaction['month_item_purchased'] =transaction['date'].dt.month

transaction['quarter_item_purchased'] = transaction['date'].dt.quarter

transaction['year_item_purchased'] = transaction['date'].dt.year

print (transaction.tail(2))
plt.figure(figsize=(25,25))

plt.plot(transaction['date'],transaction['transactions'])

plt.show()

plt.figure(figsize=(8,12))

trans_day = transaction['transactions'].groupby(transaction['year_item_purchased']).sum()

trans_day.plot(kind='bar')

plt.show()
stores = pd.read_csv("../input/stores.csv")

print (stores.head())
#Lets find out number of cities in each state, which in nothing but finding out number of stores in each

#in each state.

df = stores['city'].groupby(stores['state']).count()

df.plot(kind='bar', figsize = (12,8), yticks=np.arange(min(df), max(df)+1, 1.0), title = "Number of cities in each state")

plt.show()
#Looks like onpromotion field is always NaN, if so we will get rid of that columns 

#from the training data

print(train['onpromotion'].notnull().any())

train_new=train.drop('onpromotion',axis=1)

print(train_new.tail(5))
oils = pd.read_csv("../input/oil.csv")

oils['date'] = pd.to_datetime(oils['date'], format='%Y-%m-%d')

oils['day_item_purchased'] = oils['date'].dt.day

oils['month_item_purchased'] =oils['date'].dt.month

oils['quarter_item_purchased'] = oils['date'].dt.quarter

oils['year_item_purchased'] = oils['date'].dt.year
plt.figure(figsize=(25,25))

#trans_day = transaction['transactions'].groupby(transaction['year_item_purchased']).sum()

plt.plot(oils['date'],oils['dcoilwtico'])

#trans_day.plot(kind='bar')

plt.show()