# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
#Let's explore datasets

#aisles.csv

aisles = pd.read_csv('../input/aisles.csv')

#departments.csv

departments = pd.read_csv('../input/departments.csv')

#order_products__prior.csv

order_products_prior = pd.read_csv("../input/order_products__prior.csv")

#order_products__train.csv

order_products_train = pd.read_csv("../input/order_products__train.csv")

#orders.csv

orders = pd.read_csv("../input/orders.csv")

#products.csv

products = pd.read_csv('../input/products.csv')

#sample_submission.csv

sample_submission = pd.read_csv('../input/sample_submission.csv')
aisles.head()
departments.head()
order_products_prior.head()
order_products_train.head()
orders = orders.fillna(-1)

orders.head()


#unique DOW values

orders.order_dow.unique()

#what is the frequency of the orders according to days

n, bins, patches = plt.hist(orders.order_dow, 13, facecolor="red", alpha=.75, align='mid')

plt.xlabel("Day of Week")

plt.ylabel("Orders Count")

plt.title("When do people buy?")

plt.show()

orders.order_hour_of_day.unique()

n, bins, patches = plt.hist(orders.order_hour_of_day, 47, facecolor="red", alpha=.75, align='mid')

plt.xlabel("Hour of Day")

plt.ylabel("Orders Count")

plt.title("When do people buy in a Day?")

plt.show()
order_products_cart_prior = order_products_prior.groupby('order_id')

n_items = order_products_cart_prior['add_to_cart_order'].max()

n, bins, patches = plt.hist(n_items, 100, facecolor="red", alpha=.75, align='mid')

plt.show()
order_products_cart_prior.head()
order_products_cart_prior = order_products_train.groupby('order_id')

n_items_train = order_products_cart_prior['add_to_cart_order'].max()

n, bins, patches = plt.hist(n_items_train, 100, facecolor="red", alpha=.75, align='mid')

plt.show()
products.head()
sample_submission.head()
tmp = order_products_prior.groupby('product_id')['add_to_cart_order'].count().to_frame()

tmp.head()
merged = pd.merge(products, pd.DataFrame(tmp, columns=['add_to_cart_order']), 

                  left_on='product_id', right_index=True)

merged.head()
merged.sort_values('add_to_cart_order', ascending=False)
reordered_products_train = pd.DataFrame({'count' : order_products_train.groupby( ['reordered'] ).size()}).reset_index()

newcol = reordered_products_train['count']/reordered_products_train['count'].sum()

reordered_products_train = reordered_products_train.assign(proportion = newcol )

reordered_products_train

#tmp_reorder = order_products_train.groupby('reordered')['order_id'].count().to_frame()

#tmp_reorder.reset_index()
most_reordered_product = pd.DataFrame({'product_id':order_products_train['product_id'],

                                     'proportion':order_products_train.groupby('product_id')['reordered'].mean(),

                                      'n': order_products_train.groupby('product_id')['reordered'].count()})

merged_reordered = pd.merge(products, pd.DataFrame(most_reordered_product, columns=['n', 'product_id', 'proportion']), 

                  left_on='product_id', right_index=True).reset_index()

merged_reordered.head()

merged_reordered = merged_reordered[merged_reordered['n'] > 40.0]

merged_reordered.sort_values('proportion', ascending=False).head()
product_by_id_order = pd.DataFrame({'count':order_products_train.groupby(['product_id', 'add_to_cart_order']).size()})

new_col = product_by_id_order['count']/product_by_id_order['count'].sum()

product_by_id_order = product_by_id_order.assign(pct=new_col)

product_by_id_order = product_by_id_order.reset_index()

product_by_id_order.head()
merged_product_reordered = pd.merge(products, product_by_id_order,

                                   on='product_id', how="left")

merged_product_reordered.reset_index()

merged_product_reordered.head()

merged_product_reordered = merged_product_reordered[merged_product_reordered['add_to_cart_order'] == 1]

merged_product_reordered = merged_product_reordered[merged_product_reordered['count'] > 10]

merged_product_reordered.sort_values('pct', ascending=False, axis=0)
order_time_join = pd.merge(order_products_train, orders, on="order_id")

order_time_join.head()
order_time_groupby_reorder = pd.DataFrame({'mean_reorder': order_time_join.groupby(['days_since_prior_order'])['reordered'].mean()})

order_time_groupby_reorder = order_time_groupby_reorder.reset_index().head()

plt.ylabel('Mean Reorder')

plt.xlabel('Days Since Prior Order')

plt.bar(order_time_groupby_reorder['days_since_prior_order'], 

        order_time_groupby_reorder['mean_reorder'], color='red')

plt.show()
product_by_reordered = pd.DataFrame({'mean_reorder': order_products_train.groupby(['product_id'])['reordered'].mean(),

                                    'count': order_products_train.groupby(['product_id'])['order_id'].count()})

product_by_reordered = product_by_reordered.reset_index()
plt.xlim([0,2000])

plt.yticks([0,0.5,1.0,1.5])

plt.plot(product_by_reordered['count'],product_by_reordered['mean_reorder'] , 'ro')

plt.show()

order_groupby_orderid = pd.DataFrame({'order_id': order_products_prior.order_id,

                                      'mean_reorder': order_products_prior.groupby(['order_id'])['reordered'].mean(),

                                     'product_id':order_products_prior.product_id

                                     })

order_groupby_orderid.dtypes
order_groupby_orderid.head()
#order_groupby_orderid = order_groupby_orderid.dropna()

order_groupby_orderid.count()
# Let's Join the table to add user_id info 

merged_df = pd.merge(order_groupby_orderid, pd.DataFrame(orders, columns=['order_number', 'user_id']), 

                  left_on='product_id', right_index=True)
merged_df.reset_index()

merged_df.head()
merged_df = merged_df.dropna()

merged_df = merged_df[merged_df['mean_reorder']==1]

merged_df = merged_df[merged_df['order_number']>2]
merged_df.head()
user_id_with_meanreorder = merged_df.groupby(['user_id']).count()
#user_id_with_meanreorder.head(20)

user_id_with_meanreorder.sort_values(by='mean_reorder', ascending=False)
newcol = user_id_with_meanreorder['mean_reorder']/user_id_with_meanreorder['order_number']

user_id_with_meanreorder = user_id_with_meanreorder.assign(percent_equal = newcol)
user_id_with_meanreorder = user_id_with_meanreorder.reset_index()
user_id_with_meanreorder.head()
user_id_with_meanreorder.sort_values(by='mean_reorder', ascending=False)