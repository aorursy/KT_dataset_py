# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
order_products_train = pd.read_csv("../input/order_products__train.csv")

order_products_prior = pd.read_csv("../input/order_products__prior.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

aisles = pd.read_csv("../input/aisles.csv")

departments = pd.read_csv("../input/departments.csv")
order_products_train.head()
order_products_prior.head()
orders.head()
products.head()
aisles.head()
departments.head()
print(orders["eval_set"].value_counts())

print(orders["eval_set"].value_counts(normalize = True))
sns.countplot(x="order_dow",data = orders)

plt.xlabel("Day of week")

plt.ylabel("orders count")

plt.title("Frequency of Orders by days of week")

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="order_hour_of_day",data = orders)

plt.xlabel("Hours")

plt.ylabel("orders count")

plt.title("Frequency of Orders by hours of the day")

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x="days_since_prior_order",data=orders)

plt.title("No. of orders by days since ordered previously")

plt.xlabel("Orders count")

plt.ylabel("Days since prior order")

plt.xticks(rotation="vertical")

plt.show()
print("The ratio of products reordered in order_products_prior: {}".format(order_products_prior["reordered"].sum()/order_products_prior.shape[0]))

print("The ratio of products reordered in order_products_train: {}".format(order_products_train["reordered"].sum()/order_products_train.shape[0]))
count = order_products_train.groupby("order_id")["add_to_cart_order"].aggregate("count").reset_index()

count = count["add_to_cart_order"].value_counts()

plt.figure(figsize=(12,8))

sns.barplot(count.index,count.values)

plt.xlabel("Number of items in particular order")

plt.ylabel("Number of such orders")

plt.xticks(rotation="vertical")

plt.show()
order_products_prior = pd.merge(order_products_prior,products, on = 'product_id', how='left')

order_products_prior = pd.merge(order_products_prior,departments, on = "department_id", how="left")

order_products_prior = pd.merge(order_products_prior, aisles, on='aisle_id', how='left')

count = order_products_prior["department"].value_counts()

plt.figure(figsize=(12,8))

sns.barplot(count.index,count.values)

plt.xlabel("Departments")

plt.ylabel("Count")

plt.xticks(rotation="vertical")

plt.show()
count = order_products_prior['aisle'].value_counts().head(30)

plt.figure(figsize=(12,8))

sns.barplot(count.index, count.values)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Aisle', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
count = order_products_prior.groupby("department")["reordered"].aggregate("mean").reset_index()



print(count.head())

plt.figure(figsize=(12,8))

sns.barplot(count["department"].values, count["reordered"].values)

plt.xticks(rotation = "vertical")

plt.xlabel("Deparments", fontsize=12)

plt.ylabel("Reorder ratio", fontsize=12)

plt.ylim(0.3,0.7)

plt.show()
order_products_train = pd.merge(order_products_train, orders, on = 'order_id', how= 'left')

count = order_products_train.groupby("order_dow")["reordered"].aggregate("mean").reset_index()



print(count.head())

plt.figure(figsize=(12,8))

sns.barplot(count["order_dow"].values, count["reordered"].values)

plt.xticks(rotation = "vertical")

plt.xlabel("Day of order", fontsize=12)

plt.ylabel("Reorder ratio", fontsize=12)

plt.ylim(0.5,0.7)

plt.show()
count = order_products_train.groupby("order_hour_of_day")["reordered"].aggregate("mean").reset_index()



print(count.head())

plt.figure(figsize=(12,8))

sns.barplot(count["order_hour_of_day"].values, count["reordered"].values)

plt.xticks(rotation = "vertical")

plt.xlabel("Day of order", fontsize=12)

plt.ylabel("Reorder ratio", fontsize=12)

plt.ylim(0.5,0.7)

plt.show()