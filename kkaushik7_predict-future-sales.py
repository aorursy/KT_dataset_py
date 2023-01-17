# Exploratory data analysis of time series data in the predict future sales kaggle competition

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA



print(os.listdir("../input"))

plt.rcParams['text.color'] = 'white'

plt.rcParams['xtick.color'] = 'white'

plt.rcParams['ytick.color'] = 'white'

plt.rcParams['axes.labelcolor'] = 'white'

#{'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white'}
def plot_line(x, y, xlabel, ylabel, title, color = '#1f77b4', scatter = False):

    if (scatter):

        plt.scatter(x,y, color=color)

    else:

        plt.plot(x,y, color=color)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)
def plot_bar(x, y, xlabel, ylabel, color = '#1f77b4'):

    plt.figure(figsize=(16,8))

    plt.bar(x,y, color = color)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(xlabel + ' Vs ' + ylabel)
# Parse the date field on the training data

dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')



# Read the train and test data data

train = pd.read_csv("../input/sales_train.csv", parse_dates = ['date'], date_parser = dateparse)

test = pd.read_csv("../input/test.csv")



# Read the supplementary data

items = pd.read_csv("../input/items.csv")

item_categories = pd.read_csv("../input/item_categories.csv")

shops = pd.read_csv("../input/shops.csv")
# Add one additional feature to the training data i.e. if the sales was made on a weekday

is_weekend = lambda x: 1 if x.weekday() >= 5 else 0

train['weekend_yn'] = train['date'].apply(is_weekend)

train.head(n=10)
"""

Aggregate the data. For every month, shop and item, aggregate the following

- Number of days on which sales were made

- Number of weekends on which sales were made (and hence the proportion)

- Sum total of items sold

- Average price of the item

""" 

agg_train = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'date':'count', 'weekend_yn':np.sum, 'item_cnt_day':np.sum, 'item_price':np.mean})

agg_train.reset_index(inplace = True)

agg_train.columns = ['date_block_num', 'shop_id', 'item_id', 'num_purchase_days', 'num_weekend_days', 'item_cnt_sum', 'item_price_avg']

agg_train.head(n=10)
num_weekend = agg_train.groupby('date_block_num').agg({'num_purchase_days':np.sum, 'num_weekend_days':np.mean, 'item_cnt_sum':np.sum})

num_weekend.reset_index(inplace=True)

num_weekend.columns = ['month', 'num_purchases', 'num_weekend_purchases', 'item_count']

num_weekend.head()
plot_line(num_weekend.month, num_weekend.num_purchases, 'Month', 'Number of Purchases', 'Month Vs Number of Distinct Purchases', 'red')
plot_line(num_weekend.month, num_weekend.num_weekend_purchases, 'Month', 'Number of Weekend Purchases', 'Month Vs Number of Weekend Purchases')
plot_line(num_weekend.month, num_weekend.item_count, 'Month', 'Item Count', 'Month Vs Item Count', 'green')
plot_line(num_weekend.num_purchases, num_weekend.item_count, 'Number of purchases', 'Item Count', 'Purchase Count Vs Item Count', 'gold', True)
item_cat = pd.DataFrame(items.groupby(['item_id', 'item_category_id']).size())

item_cat.reset_index (inplace= True)

item_cat.columns = ['item_id','item_category_id', 'size']

item_cat.head()
newtrain = agg_train.merge(item_cat[['item_category_id', 'item_id']], on = 'item_id', how = 'left')

newtrain.head()
item_cat = newtrain.groupby('item_category_id').agg({'item_price_avg': np.mean, 'item_cnt_sum': np.sum})

item_cat.reset_index(inplace = True)

item_cat.head()
plot_bar(item_cat.item_category_id, item_cat.item_price_avg, 'Item Category Id', 'Average Item Price')
plot_bar(item_cat.item_category_id, item_cat.item_cnt_sum, 'Item Category Id', 'Total Items Sold', 'green')
newtrain = num_weekend

newtrain['sales'] = num_weekend['item_count'].diff(periods = 12)

#newtrain['sales'].fillna(value = np.mean(num_weekend['item_count']), inplace = True)

newtrain.dropna(inplace=True)

plot_line(newtrain.month, newtrain.sales, 'Month', 'Sales', 'Month Vs Sales', 'Red')

newtrain.head(n=8)
plot_acf(newtrain.sales)
plot_pacf(newtrain.sales)
model = ARIMA(newtrain.sales, order=(1,0,1))

model_fit = model.fit()

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)

residuals.plot(kind='kde')

plt.show()
preds = model_fit.fittedvalues
plt.plot(newtrain.sales)

plt.plot(preds, color = 'red')
item_categories.head()