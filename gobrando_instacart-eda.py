import numpy as np 
import pandas as pd 
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from subprocess import check_output
print(check_output(['ls', '../input']).decode('utf8'))
import glob, re
dfs = {re.search('/([^/\.]*)\.csv', fn).group(1):
      pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}
for k, v in dfs.items(): locals()[k] = v
opp = order_products__prior
opp.tail()
opt = order_products__train
opt.tail()
opa = pd.concat([opp, opt], axis = 0) # 'order products all'
opa.tail()
# sanity check
print(len(opp))
print(len(opt))
print(len(opa))
print(opa.isnull().sum())
order_total = opa.groupby('order_id')['add_to_cart_order'].max().reset_index()
order_total = order_total.add_to_cart_order.value_counts()[:20] #nlargest(20)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation = 'vertical')
sns.barplot(order_total.index, order_total.values)

plt.xlabel('Number of Orders', fontsize = 14) 
plt.ylabel('Number of Products in Order', fontsize = 14)
#plt.xlabel('Number of Products in Order', fontsize=14)
plt.show()
pop = opa.groupby('product_id')['add_to_cart_order'].aggregate({'total_ordered': 'count'}).reset_index()
pop = pop.merge(products[['product_id', 'product_name']], how = 'left', on = ['product_id'])
pop = pop.sort_values(by = 'total_ordered', ascending = False)[:10]
pop
pop = pop.groupby(['product_name']).sum()['total_ordered'].sort_values(ascending = False)

f, ax = plt.subplots(figsize=(12,10))
sns.set_style('darkgrid')
sns.barplot(pop.index, pop.values)

plt.xticks(rotation = 'vertical')
plt.ylabel('Number of Orders', fontsize = 14)
plt.xlabel('Most Popular Products', fontsize = 14)
plt.show()
reorder_ratio = opa.groupby('reordered')['product_id'].agg({'total_products': 'count'}).reset_index()
reorder_ratio['ratio'] = reorder_ratio['total_products'].apply(lambda x: x / reorder_ratio['total_products'].sum())
reorder_ratio
reorder_ratio = reorder_ratio.groupby(['reordered']).sum()['total_products'].sort_values(ascending = False)

f, ax = plt.subplots(figsize = (5,8))
sns.set_style('whitegrid')
sns.barplot(reorder_ratio.index, reorder_ratio.values, palette = 'RdBu')

plt.xlabel('New Item/Reordered', fontsize = 14)
plt.ylabel('Total Number of Orders', fontsize = 14)
plt.ticklabel_format(style = 'plain', axis = 'y')
plt.show()

product_rr = opa.groupby('product_id')['reordered'].agg({'reorder_total': sum, 'order_total': 'count'}).reset_index()
product_rr['reorder_probability'] = product_rr['reorder_total'] / product_rr['order_total']
product_rr = product_rr.merge(products[['product_name', 'product_id']], how = 'left', on = 'product_id')
product_rr = product_rr[product_rr.order_total > 75].sort_values(['reorder_probability'], ascending = False)[:10]
product_rr
product_rr = product_rr.sort_values('reorder_probability', ascending = False)

plt.subplots(figsize = (12, 10))
sns.set_style('darkgrid')
sns.barplot(product_rr.product_name, product_rr.reorder_probability)
plt.ylim([.85, .95])
plt.xticks(rotation = 'vertical')
plt.xlabel('Most Reordered Products')
plt.ylabel('Reorder Probability')
plt.show()
time_of_day = orders.groupby('order_hour_of_day')['order_id'].agg('count').reset_index()

f, ax = plt.subplots(figsize = (15, 10))
sns.barplot(time_of_day['order_hour_of_day'], time_of_day['order_id'])
plt.xlabel('Orders by Hour', fontsize = 14)
plt.ylabel('Total Orders', fontsize = 14)
plt.show()
grouped = orders.groupby('order_dow')['order_id'].agg('count').reset_index()

f, ax = plt.subplots(figsize = (15,10))
sns.set_style('whitegrid')
current_palette = sns.color_palette('colorblind')
sns.set_palette(current_palette)
sns.barplot(x = 'order_dow', y = 'order_id', data = grouped)
plt.xlabel('Day of Week')
plt.ylabel('Total Orders')
plt.show()
grouped = orders.groupby('days_since_prior_order')['order_id'].agg('count').reset_index()

from matplotlib.ticker import FormatStrFormatter
f, ax = plt.subplots(figsize = (15,19))
sns.barplot(x = 'days_since_prior_order', y = 'order_id', data = grouped)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xlabel('Days to Reorder')
plt.ylabel('Total Orders')
plt.show()
grouped = orders.groupby('user_id')['order_id'].agg('count')
grouped = grouped.value_counts()

f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(grouped.index, grouped.values)
plt.xlabel('Number of Orders')
plt.ylabel('Number of Customers')
plt.show()
items = products.merge(departments, how='left', on='department_id')
items = items.merge(aisles, how='left', on='aisle_id')
items.head()
grouped = items.groupby('department')['department_id'].agg({'total_products': 'count'}).reset_index()
grouped['percent_of_inv'] = grouped['total_products'].apply(lambda x: x / grouped['total_products'].sum())
grouped = grouped.sort_values(by='total_products', ascending=False)
grouped.head()
f, ax = plt.subplots(figsize=(12,10))
sns.barplot(x='department', y='total_products', data=grouped)
plt.xticks(rotation='vertical')
plt.xlabel('Department', fontsize=14)
plt.ylabel('Number of Products', fontsize=14)
plt.show
g1 = items.groupby(["department", "aisle"], as_index=False).size().reset_index(name="count")
g2 = g1.loc[g1['department'] == 'personal care']
g2 = g2.sort_values(by='count', ascending=False)[:5]
current_palette = sns.color_palette('colorblind')
sns.set_palette(current_palette)
sns.barplot(x='aisle', y='count', data=g2)
plt.xticks(rotation='vertical')
plt.title('Personal Care Department')
plt.show()
g2 = g1.loc[g1['department'] == 'snacks']
g2 = g2.sort_values(by='count', ascending=False)[:5]
current_palette = sns.color_palette('colorblind')
sns.set_palette(current_palette)
sns.barplot(x='aisle', y='count', data=g2)
plt.xticks(rotation='vertical')
plt.title('Snacks Department')
plt.show()
grouped = items.groupby('aisle')['product_id'].agg({'total_products': 'count'}).reset_index()
grouped['ratio'] = grouped['total_products'].apply(lambda x: x /grouped['total_products'].sum())
grouped = grouped.sort_values(by='total_products', ascending=False)[:10]
grouped
f, ax = plt.subplots(figsize=(12,10))
sns.barplot(x='aisle', y='total_products', data=grouped)
plt.xticks(rotation='vertical')
plt.show()
orders.head()
orders.eval_set.value_counts()
order_products = orders[['user_id', 'order_id']].merge(opa[['order_id', 'product_id']],
                                                 how='inner', on='order_id')
order_products = order_products.merge(items, how='inner', on='product_id')

grouped = order_products.groupby('department')['order_id'].agg({'total_orders': 'count'}).reset_index()
grouped['ratio'] = grouped['total_orders'].apply(lambda x: x / grouped['total_orders'].sum())
grouped = grouped.sort_values(by='total_orders', ascending=False)
grouped
f, ax = plt.subplots(figsize=(12,10))
sns.barplot(x='department', y='total_orders', data=grouped)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xticks(rotation='vertical')
plt.xlabel('Departments', fontsize=14)
plt.ylabel('Number of Orders', fontsize=14)
plt.show()
g1 = order_products.groupby(['department', 'aisle'], as_index=False).size().reset_index(name="count")
g2 = g1.loc[g1['department'] == 'produce']
g2 = g2.sort_values(by='count', ascending=False)[:5]
g2
current_palette = sns.color_palette('colorblind')
sns.set_palette(current_palette)
sns.barplot(x='aisle', y='count', data=g2)
plt.xticks(rotation='vertical')
plt.title('Top Produce Aisles')
plt.show()
grouped = order_products.groupby("aisle")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped['Ratio'] = grouped["Total_orders"].apply(lambda x: x /grouped['Total_orders'].sum())
grouped.sort_values(by='Total_orders', ascending=False, inplace=True )
grouped.head(10)
grouped  = grouped.groupby(['aisle']).sum()['Total_orders'].sort_values(ascending=False)[:15]

f, ax = plt.subplots(figsize=(12, 15))
plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.show()