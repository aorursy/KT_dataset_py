import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
%%time



aisles = pd.read_csv("../input/aisles.csv")

depts = pd.read_csv("../input/departments.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

prior = pd.read_csv("../input/order_products__prior.csv")

train = pd.read_csv("../input/order_products__train.csv")
print("length of aisles", len(aisles))

print("length of depts", len(depts))

print("length of depts", len(orders))

print("length of products", len(products))

print("length of prior", len(prior))

print("length of train", len(train))





#Missing Values Check
#orders

total = orders.isnull().sum().sort_values(ascending=True)

percentage = orders.isnull().sum()/orders.isnull().count().sort_values(ascending=True)

table = pd.concat([total,percentage],keys=['total','percentage'],axis=1)

table
#train

total = train.isnull().sum().sort_values(ascending=True)

percentage = train.isnull().sum()/train.isnull().count().sort_values(ascending=True)

table2 = pd.concat([total,percentage],keys=['total','percentage'],axis=1)

table2
#Prior

total = prior.isnull().sum().sort_values(ascending=True)

percentage = prior.isnull().sum()/prior.isnull().count().sort_values(ascending=True)

table3 = pd.concat([total,percentage],keys=['total','percentage'],axis=1)

table3
grouped = prior.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()

grouped = grouped.add_to_cart_order.value_counts()



f, ax = plt.subplots(figsize=(12.5, 8))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values,palette="cubehelix")

plt.ylabel('Number of Occurrences Prior Data', fontsize=17)

plt.xlabel('Number of products ordered in the order_id', fontsize=13)

plt.show()
grouped = train.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()

grouped = grouped.add_to_cart_order.value_counts()

f, ax = plt.subplots(figsize=(12.5, 8))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values,palette="BuGn_r")

plt.ylabel('Number of Occurrences Train Data', fontsize=17)

plt.xlabel('Number of products ordered in the order_id', fontsize=13)

plt.show()
grouped = train.groupby("product_id")["reordered"].aggregate({'count': 'count'}).reset_index()

grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])

grouped.sort_values(by='count', ascending=False, inplace=True)

grouped = grouped.head(30)

f, ax = plt.subplots(figsize=(12.5, 8))

sns.barplot(x='count', y='product_name', data=grouped.reset_index(),color='#2ecc71')

plt.ylabel('Most Ordered Products', fontsize=12)

plt.xlabel('Count Mean', fontsize=12)

plt.show()
grouped = orders.groupby('order_id')['order_hour_of_day'].aggregate("sum").reset_index()

grouped = grouped.order_hour_of_day.value_counts()

f, ax = plt.subplots(figsize=(12.5, 8))



sns.barplot(grouped.index,grouped.values,palette='GnBu_d')

plt.ylabel('Order Count', fontsize=12)

plt.xlabel('Timing 24 hours', fontsize=12)

plt.show()
grouped = orders.groupby('order_id')['order_dow'].aggregate("sum").reset_index()

grouped = grouped.order_dow.value_counts()

f, ax = plt.subplots(figsize=(12.5, 8))



sns.barplot(grouped.index,grouped.values,palette='GnBu_d')

plt.ylabel('Order Count', fontsize=12)

plt.xlabel('Days In Weaks', fontsize=12)

plt.show()
grouped = orders.groupby('order_id')['days_since_prior_order'].aggregate("sum").reset_index()

grouped = grouped.days_since_prior_order.value_counts()

f, ax = plt.subplots(figsize=(12.5, 8))

sns.barplot(grouped.index,grouped.values,palette='GnBu_d')

plt.ylabel('Order Count', fontsize=12)

plt.xlabel('Days In Weaks', fontsize=12)

plt.show()
print("Number of unique customers in the whole dataset : ",len(set(orders.user_id)))
grouped = orders.groupby("eval_set")["user_id"].apply(lambda x: len(x.unique()))



plt.figure(figsize=(7,8))

sns.barplot(grouped.index, grouped.values, palette='GnBu_d')

plt.ylabel('Number of users', fontsize=12)

plt.xlabel('Eval set', fontsize=12)

plt.title("Number of unique customers in each dataset")

plt.show()


grouped = orders.groupby('user_id')['order_id'].apply(lambda x: len(x.unique())).reset_index()

grouped = grouped.groupby('order_id').aggregate("count")

sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})

sns.set_style("whitegrid")

f, ax = plt.subplots(figsize=(10, 9))

sns.barplot(grouped.index, grouped.user_id)

plt.ylabel('Numbers of Customers')

plt.xlabel('Number of Orders per customer')

plt.xticks(rotation='vertical')

plt.show()

items  = pd.merge(left =pd.merge(left=products, right=depts, how='left'), right=aisles, how='left')

items.head()
group = items.groupby('department')["product_id"].aggregate({'Total_products': 'count'}).reset_index()

group.head
grouped = group.groupby('department').sum()['Total_products'].sort_values(ascending=False)

grouped
grouped.plot()