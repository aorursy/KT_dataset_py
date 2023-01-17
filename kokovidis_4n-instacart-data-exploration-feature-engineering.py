import pandas as pd               # for data manipulation

import matplotlib.pyplot as plt   # for plotting 

import seaborn as sns             # an extension of matplotlib for statistical graphics
orders = pd.read_csv('../input/orders.csv' )

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

products = pd.read_csv('../input/products.csv')
prd = pd.merge(orders, order_products_prior, on='order_id', how='inner')

prd.head(100)
prd[prd.user_id==1].head(45)
prd['order_number_back'] = prd.groupby('user_id')['order_number'].transform(max) - prd.order_number +1 

prd.head(15)
prd[prd.user_id==30].head(10)
prd5 = prd[prd.order_number_back <= 5]

prd5.head(15)
last_five = prd5.groupby(['user_id','product_id'])[['order_id']].count()

last_five.columns = ['times_last5']

last_five.head(10)
last_five['times_last5_ratio'] = last_five.times_last5 / 5

last_five.head(10)
prd[prd.user_id==5]
#Solution 1.1

#prd5_with_five_orders = prd5.groupby('user_id').filter(lambda x: x.order_number_back.max() == 5)

#Solution 1.2

#prd5_with_five_orders = prd5.groupby('user_id').filter(lambda x: x.order_number_back.max() > 4)
#Solution 2.1

#prd5_with_five_orders = prd5.groupby('user_id').filter(lambda x: x.order_number.max() == 5)

#Solution 2.1

#prd5_with_five_orders = prd5.groupby('user_id').filter(lambda x: x.order_number_back.max() > 4)
# Solution 3

prd5_with_five_orders = prd5.groupby('user_id').filter(lambda x: x.order_id.nunique() == 5)
#sanity check

prd5_with_five_orders[prd5_with_five_orders.user_id==5]
last_five_top = last_five[last_five.times_last5_ratio == 1].groupby('product_id')[['times_last5_ratio']].count()

last_five_top.columns = ['total_users']

last_five_top.head()
last_five_top = last_five_top.sort_values(by='total_users', ascending=False)

last_five_top = last_five_top.iloc[0:20]

last_five_top = last_five_top.reset_index()

last_five_top
last_five_top_names = pd.merge(last_five_top, products, how='left')

last_five_top_names
plt.figure(figsize=(12,8))

sns.barplot(last_five_top_names.total_users, last_five_top_names.product_name)

# add label to x-axis

plt.xlabel('Number of users', size=15)

# keep y-axis free of label

plt.ylabel('  ')

#put a title

plt.title('Top 20 products that have been ordered by most users on their last 5 orders ', size=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.show()
times = prd.groupby(['user_id', 'product_id'])[['order_id']].count()

times.columns = ['Times_Bought_N']

times.head()
total_orders = prd.groupby('user_id')[['order_number']].max()

total_orders.columns = ['total_orders']

total_orders.head()
first_order_number = prd.groupby(['user_id', 'product_id'])[['order_number']].min()

first_order_number.columns = ['first_order_number']

first_order_number.head()
first_order_number_reset = first_order_number.reset_index()

first_order_number_reset.head()
span = pd.merge(total_orders, first_order_number_reset, on='user_id', how='right')

span.head(20)
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1

span.head(30)
order_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')

order_ratio.head()
order_ratio['Order_Ratio_user_id_X_product_id'] = order_ratio.Times_Bought_N / order_ratio.Order_Range_D

order_ratio.head()
plt.figure(figsize=(15,5))

order_ratio[order_ratio.product_id == 24852].Order_Ratio_user_id_X_product_id.hist(bins=50)

plt.xlabel('order_ratio', size=10)

plt.ylabel('Number of customers')

plt.title('The distribution of order_ratio for bananas', size=10)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.show()
plt.figure(figsize=(15,5))

order_ratio[order_ratio.product_id == 24852].Order_Ratio_user_id_X_product_id.hist(cumulative=True, bins=50)

plt.xlabel('order_ratio', size=10)

plt.ylabel('Number of customers')

plt.title('The CDF of order_ratio for bananas', size=10)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.show()
order_ratio = order_ratio.set_index(['user_id','product_id'])

order_ratio.head()