# Common Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
# op stands for order_products

orders_df = pd.read_csv('../input/orders.csv')

op_train_df = pd.read_csv('../input/order_products__train.csv')

op_prior_df = pd.read_csv('../input/order_products__prior.csv')

aisles_df = pd.read_csv('../input/aisles.csv')

products_df = pd.read_csv('../input/products.csv')

departments_df = pd.read_csv('../input/departments.csv')
print (orders_df.shape)

orders_df.head()
count_eval = orders_df.eval_set.value_counts()



#Plot it

plt.figure(figsize=(8,6))

sns.barplot(count_eval.index, count_eval.values, alpha = 0.9)

plt.xlabel('Eval_Set', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.show()
def unique_count(column):

    return len(np.unique(column))



grouped_df = orders_df.groupby('eval_set')['user_id'].aggregate(unique_count)
grouped_df
# Plot it

plt.figure(figsize=(8,6))

sns.barplot(grouped_df.index, grouped_df.values, alpha=0.9)

plt.title('Number of unique customers', fontsize = 12)

plt.xlabel('Eval_Set', fontsize=12)

plt.ylabel('Count', fontsize = 12)

plt.show()
count_hour = orders_df.order_hour_of_day.value_counts()



# Plot it

fig = plt.figure(figsize=(12,6))

sns.barplot(count_hour.index, count_hour.values, alpha=0.9, color='b')

plt.xlabel('order_hour_of_day', fontsize=14)

plt.ylabel('count', fontsize=14)

plt.xticks(rotation='vertical')

plt.show()
count_day = orders_df.order_dow.value_counts()



# Plot it

fig = plt.figure(figsize=(8,6))

sns.barplot(count_day.index, count_day.values, alpha=0.9, color='pink')

plt.xlabel('day_of_week', fontsize=14)

plt.ylabel('count', fontsize=14)

plt.xticks(rotation='vertical')

plt.show()
count_day = orders_df.days_since_prior_order.value_counts()



# Plot it

fig = plt.figure(figsize=(12,6))

sns.barplot(count_day.index, count_day.values, alpha=0.9, color='r')

plt.xlabel('days_since_prior_order', fontsize=14)

plt.ylabel('count', fontsize=14)

plt.xticks(rotation='vertical')

plt.show()
print (op_train_df.shape)

op_train_df.head()
grouped_df = op_train_df.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()



# Plot it

count = grouped_df.add_to_cart_order.value_counts()

fig = plt.figure(figsize=(15,8))

sns.barplot(count.index, count.values, alpha=0.9)

plt.xlabel('Numer of items in the order_id')

plt.ylabel('Number of occurences')

plt.xticks(rotation='vertical')

plt.show()
# Merge the products_df to op_train_df in order to get product names

grouped_df = pd.merge(op_train_df, products_df, on='product_id', how='left')
grouped_df.head()
# Count the number of times each product was bought

new_df = grouped_df.groupby('product_name')['product_id'].aggregate('count')
# Plotting the top 10 bestsellers

plt.figure(figsize=(8,6))

new_df.nlargest(10).plot(kind='bar', color='r')

plt.xlabel('product_name', fontsize=14)

plt.ylabel('count', fontsize=14)

plt.title('Bestsellers', fontsize=14)

plt.xticks(fontsize=12)

plt.show()
grouped_df.reordered.value_counts()
op_train_df.reordered.astype(float).sum()/op_train_df.shape[0]
# Plot it

plt.figure(figsize=(8,6))

sns.countplot(grouped_df.reordered)

plt.xlabel('reordered', fontsize=14)

plt.ylabel('count', fontsize=14)

plt.show()
new_df = grouped_df.copy()
# Count the number of times each product was reordered

product_reordered_df = new_df.groupby('product_name')['reordered'].aggregate('sum').reset_index()
# Count the number of times each product was bought

product_bought_df = grouped_df.groupby('product_name')['product_id'].aggregate('count').reset_index()
proportion_reordered_df = pd.merge(product_reordered_df, product_bought_df, on='product_name', how='left')
proportion_reordered_df['proportion_reordered'] = (proportion_reordered_df['reordered']/

                                                  proportion_reordered_df['product_id'])
proportion_reordered_df = proportion_reordered_df[proportion_reordered_df['product_id']>40]
proportion_reordered_df.drop(['reordered', 'product_id'], axis=1, inplace=True)

pr_df = proportion_reordered_df.nlargest(10, 'proportion_reordered')

pr_df
# Plot it

plt.figure(figsize=(8,6))

sns.barplot(x=pr_df.product_name, y=pr_df.proportion_reordered, color='r', alpha=0.9)

plt.xlabel('product_name', fontsize=10)

plt.xticks(rotation='vertical', fontsize=12)

plt.ylabel('proportion_reordered', fontsize=12)

plt.title('Probability of being reordered', fontsize=14)

plt.show()