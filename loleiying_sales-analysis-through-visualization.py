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

import seaborn as sns

plt.style.use("seaborn-paper")
# read data

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
# merge sales_train and items[['item_id','item_category_id']] on 'item_id'

sales_train = sales_train.merge(items[['item_id','item_category_id']], on='item_id', how='left')



# convert to datetime format

sales_train['date'] = pd.to_datetime(sales_train['date'], infer_datetime_format=True)



sales_train.head()
# merge test and items[['item_id','item_category_id']] on item_id

test = test.merge(items[['item_id','item_category_id']], on='item_id',how='left')

test.head()
# count by item_id

sales_train['item_id'].value_counts().describe().plot(kind='box',figsize=(5,5))

plt.title('Item counts distribution')
# distribution of item price

sales_train['item_price'].plot(figsize=(15,10))

plt.title("Distribution of price")
# check the outlier which has price higher than 100,000

item_id = int(sales_train.loc[sales_train['item_price'] > 100000].item_id.values)

sales_train.loc[sales_train['item_id'] == item_id]
# check if the test dataset has the same outlier item_id 

test.loc[test['item_id'] == item_id].any()
# check the price range of the items under the same category as the outlier

item_cat_id = int(sales_train.loc[sales_train['item_price'] > 100000].item_category_id.values)

sales_train.loc[sales_train['item_category_id'] == item_cat_id].item_price.plot(kind='box', figsize=(5,6))

plt.title("Price of items under the same category")

plt.ylabel("price")
# remove outlier

sales_train = sales_train.loc[sales_train.item_price < 100000].copy()
# group data by item_id

grouped = sales_train[['item_price','date','item_id','shop_id','item_cnt_day']].set_index('item_id').groupby(level='item_id')
# check the variance of the price for each item_id

price_var = grouped['item_price'].var().sort_values()

price_var = (price_var.dropna()

             .reset_index()

             .rename(columns={'item_price':'item_price_var'})

             .sort_values(by=['item_price_var','item_id'])

             .reset_index(drop=True)

             .copy()

            )

price_var.iloc[-5:]
# the distribution of the variance of the item price

ax = plt.figure(figsize=(15,10)).add_subplot(1,1,1)

log_price_var = price_var['item_price_var'].apply(lambda x: np.log(x) if x > 0 else -10) # -10 is a sentinel for the 0 variance

sns.distplot(log_price_var,bins=100)

props = {

    'title':'Distribution of log(item_price_var)',

    'xlabel':'log(item_price_var)',

    'ylabel':'% of total',

}

ax.set(**props)

#plt.title("Distribution of log(item_price_var)")

#plt.xlabel("log(item_price_var)")

#plt.ylabel("% of total")
# group by item_category

grouped_cat = sales_train.groupby(sales_train['item_category_id'])
grouped_cat['item_id'].nunique().plot(figsize=(15,10), kind='barh')

plt.title("Count of unique items in each item category")
# distribution of sales by item_category

grouped_cat['item_cnt_day'].sum().plot(figsize=(15,10))

plt.title("Sales by item category")

plt.ylabel("Units")
# comparing distribution of number of unique items and sales in each category

# normalize the number of unique items in each category to sum to 1

nunique_cat = grouped_cat['item_id'].nunique()

nunique_pct = nunique_cat.div(nunique_cat.sum(), axis=0)



# normalize the sales to sum to 1

sales_cat = grouped_cat['item_cnt_day'].sum()

sales_pct = sales_cat.div(sales_cat.sum(), axis=0)



# plot

plt.figure(figsize=(15,10))

plt.plot(nunique_pct, label="nunique_pct")

plt.plot(sales_pct, label="sales_pct")

plt.fill_between(np.arange(len(sales_pct)), sales_pct, alpha=0.5, color='orange')

plt.legend(loc="best")

plt.title("Comparison distribution of sales and number of unique items in each category")
# item price per item category

grouped_cat['item_price'].plot(figsize=(15,10))

plt.title("item price per item category")

plt.ylabel("price")
# item price per item category

plt.figure(figsize=(15,10))

grouped_cat['item_price'].min().plot(label="min")

grouped_cat['item_price'].median().plot(label="median")

grouped_cat['item_price'].max().plot(label="max")

plt.fill_between(np.arange(grouped_cat.ngroups), grouped_cat['item_price'].quantile(0.15), 

                 grouped_cat['item_price'].quantile(0.75),color='grey', alpha=0.5, label="IQR") # inter-quartile range

plt.legend(loc="best")

plt.title("Minimum, median, and maximum of the price per item category")

plt.ylabel("item price")

plt.xlabel("item category id")
# distribution of the price per item category

plt.figure(figsize=(15,10))

grouped_cat['item_price'].plot(kind='box')

#sns.boxplot(x="item_category_id",y="item_price",data=grouped_cat)

plt.title("Boxplot of the price per item category")

plt.ylabel("price")
# sales per category over different prices

plt.figure(figsize=(15,10))

sns.scatterplot(x='item_price' ,y='item_cnt_day', hue='item_category_id', data=sales_train)

plt.title("Sales per category over different prices")

plt.ylabel("Sales")

plt.xlabel("Item price")