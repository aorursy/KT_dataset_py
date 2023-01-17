# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from astropy.visualization import hist



color = sns.color_palette()



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

pd.set_option('display.float_format', lambda x: '%.3f' % x) #Limiting floats output to 3 decimal points
# Load data

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')

train = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

prior = pd.read_csv('../input/order_products__prior.csv')

print(aisles.shape)

aisles.head(3)
print(departments.shape)

departments.head(3)
print(orders.shape)

orders.head(3)
print(products.shape)

products.head(3)
print(train.shape)

train.head(3)
print(prior.shape)

prior.head(3)
# Lets merge some data frames

df_all = pd.concat([prior, train], axis=0)

print('The size of prior and train together: ', df_all.shape)

df_all = pd.merge(df_all, products, on='product_id', how='left')

df_all = pd.merge(df_all, aisles, on='aisle_id', how='left')

df_all = pd.merge(df_all, departments, on='department_id', how='left')

print('The merged data set:')

df_all.head()
nr = [len(aisles.index), len(departments .index), 

            len(train .index), len(orders .index), len(products .index), len(prior.index)]

names = ['aisles' ,'departments', 'train', 'orders', 'products', 'prior']

n_rows = pd.Series(data=nr, index=names)



plt.figure(figsize=(8,5))

sns.barplot(n_rows.index, n_rows.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Rows', fontsize=12)

plt.xlabel('Data frame', fontsize=12)

plt.title('Count of rows in each dataset', fontsize=15)

plt.show()

cnt_dpts = df_all.department.value_counts()



plt.figure(figsize=(8,5))

sns.barplot(x=cnt_dpts.index, y=cnt_dpts.values, alpha=0.8, color=color[0])

plt.ylabel('Number of ordered products', fontsize=12)

plt.xlabel('Department', fontsize=10)

plt.title('Count of ordered products per department', fontsize=12)

plt.xticks(rotation=90)

plt.show()

print('Distribution in percentages: \n', cnt_dpts/df_all.shape[0], sep='')
# create a filter function that allows for chaining

def fltr(df, key, value):

    return df[df[key] == value]

pd.DataFrame.fltr = fltr



cnt_prd = df_all.fltr('department', 'produce').aisle.value_counts()

print('Percentage of aisles in department produce:\n', cnt_prd/ sum(cnt_prd), sep='')



cnt_dry = df_all.fltr('department', 'dairy eggs').aisle.value_counts()

print('Percentage of aisles in department dairy eggs:\n', cnt_dry / sum(cnt_dry), sep='')



cnt_snks = df_all.fltr('department', 'snacks').aisle.value_counts()

print('Percentage of aisles in department snacks:\n', cnt_snks / sum(cnt_snks), sep='')
cnt_pds = pd.concat([cnt_prd, cnt_dry, cnt_snks]).sort_values(ascending=False)

print('Percentages of aisles in departments produce, dairy eggs and snacks together:\n',

     (cnt_pds / sum(cnt_pds)).head(7), sep='')
cnt_prdcts = df_all.product_name.value_counts()



print('Top 10 of ordered products (and percentage):\n')

print( (cnt_prdcts / sum(cnt_prdcts)).head(10) )
rodr = df_all.reordered.value_counts()



print('Percentage of reordered products:\n')

print(rodr.rename(index={1:'Reordered', 0:'Not reordered'}) / sum(rodr))
rodr_prdcts = df_all.fltr('reordered', 1).product_name.value_counts()



print('Percentage of most reordered products:\n')

print( (rodr_prdcts / sum(rodr_prdcts)).head(10) )
grouped = df_all.groupby('product_name')['reordered'].aggregate({'reorder_sum': sum,

    'order_total': 'count'}).reset_index()

grouped['reorder_prob'] = (grouped['reorder_sum'] + 1 ) / (grouped['order_total'] + 2 )

grouped = grouped.sort_values(['reorder_prob'], ascending=False)

grouped[grouped['order_total'] > 500].head(10)
plt.figure(figsize=(8,5))

hist(grouped.reorder_prob,  bins = 'freedman', normed=True)

plt.title('Histogram: Reorder Probability')

grouped.reorder_prob.plot(kind='density', lw=2)

plt.xlim(0,0.9)

plt.show()
orders_per_hour = orders.order_hour_of_day.value_counts()

sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(8, 5))

sns.barplot(orders_per_hour.index, orders_per_hour.values)

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Hours of order in a day', fontsize=13)

plt.show()
orders_per_wday = orders.order_dow.value_counts()

sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(8, 5))

sns.barplot(orders_per_wday.index, orders_per_wday.values)

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Weekday', fontsize=13)

plt.show()
from matplotlib.ticker import FormatStrFormatter



days_bw_reorders = orders.days_since_prior_order.value_counts()

sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(8, 5))

sns.barplot(days_bw_reorders.index, days_bw_reorders.values)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Days between reorders', fontsize=13)

plt.show()
orders_na = ( orders.days_since_prior_order.isnull().sum()  / len(orders) ) * 100

print('Percentage of orders that are first order: ', orders_na)



orders.days_since_prior_order.describe()
grouped = orders.groupby('user_id')['order_number'].aggregate(np.max).reset_index()

print('Average number of orders per customer: ', grouped.order_number.mean())

grouped = grouped.order_number.value_counts()



sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(grouped.index, grouped.values)

#ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.ylabel('Number of Occurrences', fontsize=13)

plt.xlabel('Number of orders', fontsize=13)

plt.xticks(rotation='vertical')

plt.show()
grouped = df_all.groupby('order_id')['add_to_cart_order'].aggregate(np.max).reset_index()

grouped = grouped.add_to_cart_order.value_counts()

sns.set_style('darkgrid')

plt.figure(figsize=(16,11))

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of Occurences', fontsize=12)

plt.xlabel('Items in Order', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()