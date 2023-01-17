# Numpy for numerical computing
import numpy as np

# Pandas for Dataframes
import pandas as pd
pd.set_option('display.max_columns',100)

# Matplolib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline

# Seaborn for easier visualization
import seaborn as sns

# Datetime deal with dates formats
import datetime as dt
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
items = pd.read_csv('../input/items.csv')
items_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
# Dataframe dimensions
print('The dimension of the training set is:',train.shape,'\n')
print('The feature types are:\n', train.dtypes,'\n')
print('Number of missing values:\n',train.isnull().sum())
train.head(3)
print('The dimension of the test set is:',test.shape,'\n')
print('The feature types are:\n', test.dtypes,'\n')
print('Number of missing values:\n',test.isnull().sum())
test.head(3)
print('The dimension of the items set is:',items.shape,'\n')
print('The feature types are:\n', items.dtypes,'\n')
print('Number of missing values:\n',items.isnull().sum())
items.head(3)
print('The dimension of the items categories is:',items_categories.shape,'\n')
print('The feature types are:\n', items_categories.dtypes,'\n')
print('Number of missing values:\n', items_categories.isnull().sum())
items_categories.head(3)
print('The dimension of the shops set is:',shops.shape,'\n')
print('The feature types are:\n', shops.dtypes,'\n')
print('Number of missing values:\n', shops.isnull().sum())
shops.head(3)
# Change the date type
date = train.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))

# Create 3 new features for year, month and day
train['year'] = date.dt.year
train['month'] = date.dt.month
train['day'] = date.dt.day
train.head()

# Remove the "date" feature
train = train.drop('date', axis=1)
# Add the "item_category_id" to the dataset
train = pd.merge(train, items.drop('item_name', axis=1), on='item_id')
train.head()
# Create "revenue" feature
train['revenue'] = train.item_price*train.item_cnt_day
train.head()
# Plot the total number of products sold by year
train.groupby('year').item_cnt_day.sum().plot()
plt.xticks(np.arange(2013, 2016, 1))
plt.xlabel('Year')
plt.ylabel('Total number of products sold')
plt.show()

# Plot the total number of products sold by month for each year
train.groupby(['month','year']).sum()['item_cnt_day'].unstack().plot()
plt.xlabel('Month')
plt.ylabel('Total number of products sold')
plt.show()
# Plot the total revenue by year
train.groupby('year').revenue.sum().plot()
plt.xticks(np.arange(2013, 2016, 1))
plt.xlabel('Year')
plt.ylabel('Total revenue')
plt.show()

# Plot the total revenue by month for each year
train.groupby(['month','year']).sum()['revenue'].unstack().plot()
plt.xlabel('Month')
plt.ylabel('Total revenue')
plt.show()
# Plot the top 10 items
sns.countplot(y='item_id', hue='year', data=train, order = train['item_id'].value_counts().iloc[:10].index)
plt.xlim(0,20000)
plt.xlabel('Number of times the item was sold')
plt.ylabel('Identifier of the item')
plt.show()

# Plot the top 10 shops
sns.countplot(y='shop_id', hue='year', data=train, order = train['shop_id'].value_counts().iloc[:10].index)
plt.xlabel('Number of times the shop sold')
plt.ylabel('Identifier of the shop')
plt.show()