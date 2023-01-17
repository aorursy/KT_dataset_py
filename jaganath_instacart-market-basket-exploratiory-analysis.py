# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

color = sns.color_palette()

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading the files

orders = pd.read_csv("../input/orders.csv")

departments = pd.read_csv("../input/departments.csv")

products = pd.read_csv("../input/products.csv")

order_prod_train = pd.read_csv("../input/order_products__train.csv")

order_prod_prior = pd.read_csv("../input/order_products__prior.csv")

aisles = pd.read_csv("../input/aisles.csv")

datafiles = [orders, departments, products, order_prod_train, order_prod_prior,aisles]



for var in datafiles:

    print(var.info())

    print("-"*100)

    
orders.head(10)
orders.head(40)
#order_products__prior.csv contains all the prior data

order_prod_prior.head()
#order_products__train.csv conatins all the train data

order_prod_train.head()
import IPython

 

# Grouping by one factor

df_id = orders.groupby('user_id')

 

# Getting all methods from the groupby object:

attr = [method for method in dir(df_id)

 if callable(getattr(df_id, method)) & ~method.startswith('_')]

 

# Printing the result

print(IPython.utils.text.columnize(attr))
#group by "user_id" and take the max value of "order_number" column for each group in user_id and aggregate it.

cap = orders.groupby("user_id")["order_number"].aggregate(np.max).reset_index()

cap = cap.order_number.value_counts()

sns.set_style("whitegrid")

plt.figure(figsize=(15,12))

sns.barplot(cap.index, cap.values)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Max order number', fontsize=12)

plt.xticks(rotation= 90)

plt.show()
#Let's check how many customers are there totally

print("There are {} customers".format(sum(orders.groupby("eval_set")["user_id"].nunique().values[1:])))
cap = orders.groupby("eval_set").size()

plt.figure(figsize = (10,7))

sns.barplot(cap.index, cap.values, palette = "coolwarm")

plt.ylabel("No of Orders", fontsize = 14)

plt.xlabel("Dataset",fontsize = 14)

plt.title("number of unique orders in different dataset", fontsize = 16)

plt.show()
cap = orders.groupby("order_dow")["order_id"].size()

#sns.barplot(cap.index, cap.values, color = color[9], alpha = 0.8)

plt.figure(figsize=(10,8))

ax = sns.barplot(cap.index, cap.values, alpha=0.8, color=color[9])

ax.set_xlabel('Day of the week', fontsize = 10)

ax.set_ylabel('Total number of orders placed during the day',fontsize = 10)

ax.set_xticklabels(["Sat", "Sun", "Mon","Tue","Wed","Thu","Fri"], fontsize=10)

plt.show()
cap = orders.groupby("order_hour_of_day")["order_id"].size()

#sns.barplot(cap.index, cap.values, color = color[9], alpha = 0.8)

plt.figure(figsize=(10,8))

ax = sns.barplot(cap.index, cap.values, alpha=0.8, color=color[0])

ax.set_xlabel('Hour of the Day', fontsize = 12)

ax.set_ylabel('Hourly total number of orders', fontsize = 12)

ax.set_title('Hour of the day - Order Pattern', fontsize = 14)

plt.show()
plt.figure(figsize = (12,9))

sns.countplot("days_since_prior_order",data = orders, color = color[2])

plt.ylabel('Frequency', fontsize=14)

plt.xlabel('Days since prior order', fontsize=14)

plt.xticks(rotation = 90)

plt.title("Order Patterns of customers", fontsize=15)

plt.show()
df = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].agg("count").reset_index(name = "count")

pivoted_df = df.pivot('order_dow', 'order_hour_of_day', 'count')

#pivoted_df.head()

plt.figure(figsize=(14,8))

sns.heatmap(pivoted_df)

plt.title("Frequency of Orders - Day of week Vs Hour of day")

plt.ylabel("Day of the week", fontsize = 14)

plt.xlabel("Hour of the Day", fontsize = 14)

plt.show()
concat = pd.concat([order_prod_prior, order_prod_train])
#checking and validating the concatenation of the two dataset

len(concat)
len(order_prod_prior) + len(order_prod_train)
#left joining the dataset so that none of the rows in the concat dataset gets deleted or ignored during the join

concat = pd.merge(concat, products, on='product_id', how='left')

concat = pd.merge(concat, aisles, on='aisle_id', how='left')

concat = pd.merge(concat, departments, on='department_id', how='left')
orders.head()
concat.head()
cap = concat.groupby("product_name").size().sort_values(ascending = False)[:20]

print(cap)
plt.figure(figsize= (14,8))

sns.barplot(cap.index, cap.values, color=color[1])

plt.ylabel("Frequency", fontsize = 12)

plt.xlabel("Products", fontsize = 12)

plt.xticks(rotation = 90, fontsize = 12)

plt.show()
cap = concat.groupby("aisle").size().sort_values(ascending = False)[:20]

print(cap)
plt.figure(figsize= (14,8))

sns.barplot(cap.index, cap.values, color=color[3])

plt.ylabel("Frequency", fontsize = 12)

plt.xlabel("Products", fontsize = 12)

plt.xticks(rotation = 90, fontsize = 12)

plt.show()
cap = concat.groupby("department").size().sort_values(ascending = False)[:20]

print(cap)
plt.figure(figsize= (14,8))

sns.barplot(cap.index, cap.values, color=color[8])

plt.ylabel("Frequency", fontsize = 12)

plt.xlabel("Products", fontsize = 12)

plt.xticks(rotation = 90,fontsize = 12)

plt.show()
concat.head()
items  = pd.merge(left =pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')

items.head()

group_val = items.groupby("department")["product_id"].count().sort_values(ascending = False)

plt.figure(figsize=(12,12))

labels = (np.array(group_val.index))

sizes = (np.array((group_val / group_val.sum())*100))

plt.pie(sizes, labels=labels, 

        autopct='%1.1f%%', startangle=200)

plt.title("Departments distribution", fontsize=15)

plt.show()
users_flow = orders[['user_id', 'order_id']].merge(concat[['order_id', 'product_id']],how='inner', left_on='order_id', right_on='order_id')

users_flow = users_flow.merge(items, how='inner', left_on='product_id',right_on='product_id')

grouped = users_flow.groupby("department")["order_id"].count().reset_index(name = "Total_orders")

grouped.sort_values(by = "Total_orders",ascending=False, inplace=True)

grouped = grouped.reset_index(drop = True)

plt.figure(figsize=(12,8))

sns.pointplot(grouped['department'], grouped['Total_orders'].values, alpha=0.8, color=color[0])

plt.ylabel('Sales', fontsize=12)

plt.xlabel('Department', fontsize=12)

plt.title("Departmentwise Sales", fontsize=15)

plt.xticks(rotation='vertical')

#plt.show()
orders.head()
concat.head()
concat_orders = pd.merge(orders, concat, on = "order_id", how = "right")
concat_orders.head(20)