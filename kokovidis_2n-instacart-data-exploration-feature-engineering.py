import pandas as pd               # for data manipulation

import matplotlib.pyplot as plt   # for plotting 

import seaborn as sns             # an extension of matplotlib for statistical graphics
orders = pd.read_csv('../input/orders.csv' )

products = pd.read_csv('../input/products.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')
order_products_prior.head(12)
size = order_products_prior.groupby('order_id')[['product_id']].count()

size.head(10)
size.columns= ['order_size']

size.head()
# First check the available data on order_products_prior

order_products_prior.head()
# Write your answer

size = order_products_prior.groupby('order_id')[['add_to_cart_order']].max()

size.columns= ['order_size']

size.head()
size_results = size.groupby('order_size')[['order_size']].count()

size_results.columns = ['total_orders']

size_results.head()
plt.figure(figsize=(15,10))

#size_of_order will be on our x-axis and total_orders the y-axis

graph = sns.barplot(size_results.index, size_results.total_orders)

# we modify the x-ticks

graph.set( xticks=list( range(0,size_results.index.max(),10) ), xticklabels=list( range(0,size_results.index.max(),10) ) )

plt.ylabel('Number of orders', fontsize=15)

plt.xlabel('Number of products', fontsize=15)

plt.show()
# execution time: 25 sec

# the x on lambda function is a temporary variable which represents each group

# shape[0] on a DataFrame returns the number of rows

reorder = order_products_prior.groupby('product_id').filter(lambda x: x.shape[0] >40)

reorder.head()
#execution time 30 sec

reorder = order_products_prior.groupby('product_id').filter(lambda x: x.product_id.count() >40)

reorder.head()
reorder = reorder.groupby('product_id')[['reordered']].mean()

reorder.columns = ['reorder_ratio']

reorder.head()
reorder = reorder.sort_values(by='reorder_ratio', ascending=False)

reorder_10 = reorder.iloc[0:10]

reorder_10.head(10)
plt.figure(figsize=(12,8))

sns.barplot(reorder_10.index, reorder_10.reorder_ratio, order=reorder_10.index)

plt.xlabel('10 top products \n Note that each ID corresponds to a product from products data frame', size=15)

plt.ylabel('Reorder probability', size=15)

#we set the range of y-axis to a bit lower from the lowest probability and a bit higher from the higest probability

plt.ylim(0.87,0.95)

plt.show()
products[products.product_id == 6433]
plt.hist(reorder.reorder_ratio, bins=100)

plt.show()
reorder_ratio_orders= order_products_prior.groupby('order_id')[['reordered']].mean()

reorder_ratio_orders.columns= ['reordered_ratio']

reorder_ratio_orders.head()
plt.hist(reorder_ratio_orders.reordered_ratio, bins=20)

plt.show()
# Write your code here

reorder_ratio_orders[reorder_ratio_orders.reordered_ratio== 1].count()
ratio_one_count = reorder_ratio_orders[reorder_ratio_orders.reordered_ratio== 1].count()

all_orders = reorder_ratio_orders.reordered_ratio.count()

percentage = (ratio_one_count / all_orders)*100

print('Orders with reorder ratio = 1 are ' + str(round(percentage[0], 2)) + ' % of all orders.')