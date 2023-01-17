# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import gc
color = sns.color_palette()

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

op_prior_df = pd.read_csv("../input/order_products__prior.csv")
print(op_prior_df.shape)
op_train_df = pd.read_csv("../input/order_products__train.csv")
print(op_train_df.shape)
orders_df = pd.read_csv("../input/orders.csv")
print(orders_df.shape)
orders_df.head()
orders_df.isnull().sum()
orders_df[orders_df.days_since_prior_order.isnull()==True].head()
(orders_df.loc[orders_df.days_since_prior_order.isnull()==True, \
               'order_id'] == orders_df.loc[orders_df.order_number==1, 'order_id']).all()
op_prior_df.head()
op_prior_df.isnull().sum()
op_train_df.head()
op_train_df.isnull().sum()
print(orders_df.user_id.nunique())
order_eval_cnt = orders_df.eval_set.value_counts()
print(order_eval_cnt)

# Plot it
plt.figure(figsize=(12,5))
sns.barplot(order_eval_cnt.index, order_eval_cnt.values, alpha=0.8, color=color[3])
plt.title("Count of eval_set in orders data set")
plt.xlabel("eval_set")
plt.ylabel("count")
plt.show()
plt.figure(figsize=(12,5))
orders_df.user_id.value_counts().plot.hist(alpha=0.8, color=color[3])
plt.title("Number of orders per user")
plt.xlabel("Number of orders")
plt.ylabel("Frequency")
plt.show()
op_concate_df = op_prior_df.append(op_train_df, ignore_index=True)
op_concate_df.shape
op_dedup_len = op_concate_df[['order_id','product_id']].drop_duplicates().shape[0]
print(op_dedup_len != 33819106)
order_pd_cnt = op_concate_df.groupby(['order_id']).size().value_counts()
plt.figure(figsize=(12,5))
sns.barplot(order_pd_cnt.index, order_pd_cnt.values, alpha=0.8, color=color[3])
plt.title("Number of products per order")
plt.xlabel("Number of products")
plt.ylabel("Frequency")
plt.show()
order_dow_cnt = orders_df['order_dow'].value_counts()
plt.figure(figsize=(12,5))
sns.barplot(order_dow_cnt.index, order_dow_cnt.values, alpha=0.8, color=color[3])
plt.title("Distribution of Order Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Frequency")
plt.show()
order_hour_cnt = orders_df['order_hour_of_day'].value_counts()
plt.figure(figsize=(12,5))
sns.barplot(order_hour_cnt.index, order_hour_cnt.values, alpha=0.8, color=color[3])
plt.title("Distribution of Order Hour of Day")
plt.xlabel("Our of Day")
plt.ylabel("Frequency")
plt.show()
dow_hour_df = orders_df.groupby(['order_dow','order_hour_of_day'])['order_number'].agg('count').reset_index()
dow_hour_df = dow_hour_df.pivot('order_dow','order_hour_of_day','order_number')
plt.figure(figsize=(12,8))
sns.heatmap(dow_hour_df)
days_prior_cnt = orders_df['days_since_prior_order'].value_counts()
plt.figure(figsize=(12,5))
sns.barplot(days_prior_cnt.index, days_prior_cnt.values, alpha=0.8, color=color[3])
plt.title("Distribution of Days Since Prior Order")
plt.xlabel("Days Since Prior Order")
plt.ylabel("Frequency")
plt.show()
dow_daysprior_df = orders_df.groupby(['order_dow','days_since_prior_order'])['order_number'].agg('count').reset_index()
dow_daysprior_df = dow_daysprior_df.pivot('order_dow','days_since_prior_order','order_number')
plt.figure(figsize=(12,8))
sns.heatmap(dow_daysprior_df)
del orders_df
gc.collect()
prop_reorder = op_concate_df.groupby('order_id')['reordered'].agg(['count','sum'])
prop_reorder = prop_reorder['sum'] / prop_reorder['count']
prop_reorder.hist(figsize=(12,5), color=color[5], alpha=.8)
plt.title('Distribution of proportion of reordered products')
del prop_reorder
gc.collect()
op_concate_df.add_to_cart_order.hist(color=color[5],alpha=.8,figsize=(12,5),bins=145)
plt.title('Distribution of add to cart order')
op_concate_df.add_to_cart_order.describe()
add_to_cart_order_grp = pd.qcut(op_concate_df['add_to_cart_order'], 4)
p_reord_cart_ord = op_concate_df.groupby(add_to_cart_order_grp)['reordered'].agg(['count','sum'])
p_reord_cart_ord['p_reorder'] = p_reord_cart_ord['sum'] / p_reord_cart_ord['count']
del add_to_cart_order_grp
gc.collect()
plt.figure(figsize=(12,5))
sns.barplot(p_reord_cart_ord.index, p_reord_cart_ord.p_reorder, color=color[5], alpha=.8, )
plt.title('Probability of product reordered by add to cart order')
aisles_df = pd.read_csv("../input/aisles.csv")
print(aisles_df.shape)
dpmt_df = pd.read_csv("../input/departments.csv")
print(dpmt_df.shape)
products_df = pd.read_csv("../input/products.csv")
print(products_df.shape)
products = pd.merge(left=products_df, right=dpmt_df, on='department_id', how='left')
products = pd.merge(left=products, right=aisles_df, on='aisle_id', how='left')
products.head()
del aisles_df, dpmt_df, products_df
gc.collect()
products.isnull().sum()
aisel_dpmt = products[['aisle','department']].drop_duplicates().groupby('aisle')['department'].value_counts()
aisel_dpmt.head()
(aisel_dpmt==1).all()
aisel_dpmt = pd.DataFrame(aisel_dpmt)
aisel_dpmt = aisel_dpmt.rename(columns={'department':'count'})
aisel_dpmt = aisel_dpmt.reset_index()
num_aisel_dpmt = aisel_dpmt.groupby('department').size()
num_aisel_dpmt = num_aisel_dpmt.sort_values(ascending=False)
plt.figure(figsize=(16,5))
sns.barplot(num_aisel_dpmt.index, num_aisel_dpmt.values, color=color[5], alpha=.8)
plt.xticks(rotation = 'vertical')
pd_dpmt = products[['product_name','department']].groupby('department')['product_name'].count()
pd_dpmt = pd_dpmt.sort_values(ascending=False)
plt.figure(figsize=(16,5))
sns.barplot(pd_dpmt.index, pd_dpmt.values, color=color[5], alpha=.8)
plt.xticks(rotation = 'vertical')
indexes = np.linspace(0, len(op_concate_df), num=100, dtype=np.int32)
len_op_concat = len(op_concate_df)
order_pd = pd.merge(left=op_concate_df.loc[:indexes[1],:], right=products, on='product_id', how='left')
op_concate_df = op_concate_df.loc[indexes[1]:len_op_concat,:]
%%time
for i in range(len(indexes)-2):
    temp = pd.merge(left=op_concate_df.loc[:indexes[i+2],:], right=products, on='product_id', how='left')
    if i == len(indexes)-3:
        del op_concate_df
    else:
        op_concate_df = op_concate_df.loc[indexes[i+2]:len_op_concat,:]
    order_pd = order_pd.append(temp, ignore_index=True)
order_pd.head()
bestsellers = order_pd.groupby('product_name').size()
bestsellers = bestsellers.sort_values(ascending=False)
top = 15
bestsellers = bestsellers[:top]
plt.figure(figsize=(16,5))
sns.barplot(bestsellers.index, bestsellers.values, color=color[5], alpha=.8)
plt.xticks(rotation = 'vertical')
bestsellers_dpmt = order_pd.groupby('department').size()
bestsellers_dpmt = bestsellers_dpmt.sort_values(ascending=False)
bestsellers_dpmt = bestsellers_dpmt[:top]
plt.figure(figsize=(16,5))
sns.barplot(bestsellers_dpmt.index, bestsellers_dpmt.values, color=color[5], alpha=.8)
plt.xticks(rotation = 'vertical')
bestsellers_aisle = order_pd.groupby('aisle').size()
bestsellers_aisle = bestsellers_aisle.sort_values(ascending=False)
bestsellers_aisle = bestsellers_aisle[:top]
plt.figure(figsize=(16,5))
sns.barplot(bestsellers_aisle.index, bestsellers_aisle.values, color=color[5], alpha=.8)
plt.xticks(rotation = 'vertical')
most_reorder = order_pd.groupby('product_name')['reordered'].agg(['sum','count'])
most_reorder['reordered'] = most_reorder['sum'] / most_reorder['count']
most_reorder = most_reorder.sort_values('reordered',ascending=False)
most_reorder = most_reorder[:2*top]
most_reorder
most_reorder = order_pd.groupby('product_name')['reordered'].agg(['sum','count'])
most_reorder = most_reorder[most_reorder['count'] > 10000]
most_reorder['reordered'] = most_reorder['sum'] / most_reorder['count']
most_reorder = most_reorder.sort_values('reordered',ascending=False)
most_reorder = most_reorder[:top]
fig = plt.figure(figsize=(16,5)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

most_reorder['reordered'].plot(kind='bar', color=color[5], ax=ax, width=width, position=1)
most_reorder['count'].plot(kind='bar', color=color[4], ax=ax2, width=width, position=0)

ax.set_ylabel('probability of being reordered')
ax2.set_ylabel('count of order that contains this product')
ax.set_ylim(.7,.875)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=2)

handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2, labels2, loc=0)

plt.xlabel(most_reorder.index)

plt.show()
most_reorder_dpmt = order_pd.groupby('department')['reordered'].agg(['sum','count'])
most_reorder_dpmt['reordered'] = most_reorder_dpmt['sum'] / most_reorder_dpmt['count']
most_reorder_dpmt = most_reorder_dpmt.sort_values('reordered',ascending=False)
fig = plt.figure(figsize=(16,5)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

most_reorder_dpmt['reordered'].plot(kind='bar', color=color[5], ax=ax, width=width, position=1)
most_reorder_dpmt['count'].plot(kind='bar', color=color[4], ax=ax2, width=width, position=0)

ax.set_ylabel('probability of being reordered')
ax2.set_ylabel('count of order that contains products from this department')
ax.set_ylim(.3,.7)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=2)

handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2, labels2, loc=0)

plt.xlabel(most_reorder_dpmt.index)

plt.show()
most_reorder_aisle = order_pd.groupby('aisle')['reordered'].agg(['sum','count'])
most_reorder_aisle['reordered'] = most_reorder_aisle['sum'] / most_reorder_aisle['count']
most_reorder_aisle = most_reorder_aisle.sort_values('reordered',ascending=False)
most_reorder_aisle = most_reorder_aisle[:top]
most_reorder_aisle
fig = plt.figure(figsize=(16,5)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

most_reorder_aisle['reordered'].plot(kind='bar', color=color[5], ax=ax, width=width, position=1)
most_reorder_aisle['count'].plot(kind='bar', color=color[4], ax=ax2, width=width, position=0)

ax.set_ylabel('probability of being reordered')
ax2.set_ylabel('count of order that contains products from this department')
ax.set_ylim(.6,.8)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=2)

handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2, labels2, loc=0)

plt.xlabel(most_reorder_aisle.index)

plt.show()
