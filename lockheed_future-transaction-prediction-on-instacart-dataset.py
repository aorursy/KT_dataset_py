# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
aisles_data = pd.read_csv("../input/aisles.csv")
departments_data = pd.read_csv("../input/departments.csv")
order_products_train_data = pd.read_csv("../input/order_products__train.csv")
order_products_prior_data = pd.read_csv("../input/order_products__prior.csv")
orders_data = pd.read_csv("../input/orders.csv")
products_data = pd.read_csv("../input/products.csv")
aisles_data.head(10)
departments_data.head(10)
order_products_train_data.head(10)
order_products_prior_data.head(10)

orders_data.head(15)
products_data.head(10)
cnt_srs = orders_data.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
data = orders_data.groupby("eval_set")["user_id"]
data_cnt = data.size()
data = 0
print(data_cnt)
print("Total customer: ", len(orders_data.groupby("user_id")))
transactions = orders_data.groupby("user_id")["eval_set"].size()
plt.hist(transactions)
plt.title("Number of elements for  different transactions")
plt.xlabel("# of elements")
plt.ylabel("count")
plt.show()
print("min number of element: ", min(transactions))
print("max number of element: ", max(transactions))
transactions = 0
sns.countplot(x="order_dow", data = orders_data)
plt.title("Transactions for day of week")
plt.show()
frequent_order_time_period = orders_data.groupby("days_since_prior_order")["user_id"].count()
plt.stem(frequent_order_time_period)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders_data, color=color[2])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()
grouped_data = orders_data.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_data = grouped_data.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_data)
plt.title("Frequency of Day of week Vs Hour of day", fontsize=15)
plt.show()
order_products_prior_data.reordered.sum() / order_products_prior_data.shape[0]
order_products_train_data.reordered.sum() / order_products_train_data.shape[0]
grouped_data = order_products_prior_data.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_data["reordered"].loc[grouped_data["reordered"]>1] = 1
grouped_data.reordered.value_counts() / grouped_data.shape[0]
grouped_data = order_products_train_data.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_data["reordered"].loc[grouped_data["reordered"]>1] = 1
grouped_data.reordered.value_counts() / grouped_data.shape[0]
grouped_data = order_products_train_data.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_data.add_to_cart_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of products in the given order', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
order_products_train__data = pd.merge(order_products_train_data, products_data, on='product_id', how='left')
order_products_train__data = pd.merge(order_products_train__data, aisles_data, on='aisle_id', how='left')
order_products_train__data = pd.merge(order_products_train__data, departments_data, on='department_id', how='left')
order_products_train__data.head()
cnt_srs = order_products_train__data['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
cnt_srs
cnt_srs = order_products_train__data['aisle'].value_counts().head(20)
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Aisle', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
plt.figure(figsize=(10,10))
temp_series = order_products_train__data['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()
grouped_data = order_products_train__data.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_data['department'].values, grouped_data['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
grouped_data = order_products_train__data.groupby(["department_id", "aisle"])["reordered"].aggregate("mean").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_data.reordered.values, grouped_data.department_id.values)
for i, txt in enumerate(grouped_data.aisle.values):
    ax.annotate(txt, (grouped_data.reordered.values[i], grouped_data.department_id.values[i]), rotation=45, ha='center', va='center', color='green')
plt.xlabel('Reorder Ratio')
plt.ylabel('department_id')
plt.title("Reorder ratio of different aisles", fontsize=15)
plt.show()
order_products_train__data["add_to_cart_order_mod"] = order_products_train__data["add_to_cart_order"].copy()
order_products_train__data["add_to_cart_order_mod"].loc[order_products_train__data["add_to_cart_order_mod"]>70] = 70
grouped_data = order_products_train__data.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_data['add_to_cart_order_mod'].values, grouped_data['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
order_products_train__data = pd.merge(order_products_train__data, orders_data, on='order_id', how='left')
grouped_data = order_products_train__data.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()
grouped_data.head()
plt.figure(figsize=(12,8))
sns.barplot(grouped_data['order_dow'].values, grouped_data['reordered'].values, alpha=0.8, color=color[3])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Reorder ratio across day of week", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()