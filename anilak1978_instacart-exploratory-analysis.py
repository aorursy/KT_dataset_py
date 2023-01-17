# import required packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# load the data to dataframes

orders=pd.read_csv("../input/instacart/orders.csv")

products=pd.read_csv("../input/instacart/products.csv")

aisles=pd.read_csv("../input/instacart/aisles.csv")

departments=pd.read_csv("../input/instacart/departments.csv")

order_products_prior=pd.read_csv("../input/instacart/order_products__prior.csv")
# quick overview of all the orders

orders.head()
orders.info()
orders.isnull().sum()
# quick overview of the products

products.head()
products.info()
products.isnull().sum()
aisles.head()
aisles.info()
aisles.isnull().sum()
# merge aisles and products

products=pd.merge(aisles, products, on="aisle_id")

products.head()
# quick overview of departments 

departments.head()
departments.info()
departments.isnull().sum()
# merge departments and products

products=pd.merge(departments, products, on="department_id")

products.head()
# quick overview of orders prior to the most recent user order

order_products_prior.head()
order_products_prior.info()
order_products_prior.isnull().sum()
# merge order_products_prior with products dataframe

products=pd.merge(order_products_prior, products, on="product_id")

products.head()
# merge orders and products with order_id common

products_and_orders=pd.merge(products, orders, on="order_id")

products_and_orders.head()
orders.head()
# look at orders per customer

orders_per_customer=orders.groupby("user_id")["order_number"].max().reset_index()

orders_per_customer.head()
orders_per_customer["order_number"].value_counts()
plt.figure(figsize=(25,15))

sns.countplot(orders_per_customer["order_number"])

plt.title("Number of Orders Vs Number of Customer Makes These Orders")

plt.xlabel("Number of Orders By Customers")

plt.ylabel("Number of Customers")
# Look at which day of the week customers order

orders_dow = orders["order_dow"].value_counts()

orders_dow
plt.figure(figsize=(20,15))

sns.countplot(orders["order_dow"])

plt.title("Purchase Day of the Week Distribution")

plt.xlabel("Day of the week")

plt.ylabel("Count")
# Purchase hour of the day

orders_how=orders["order_hour_of_day"].value_counts()

orders_how
plt.figure(figsize=(15,10))

sns.countplot(orders["order_hour_of_day"])

plt.title("Purchase Hour Distribution")

plt.xlabel("Hour of the Day")

plt.ylabel("Count")
# frequency of orders

order_frequency=orders.groupby("order_id")["days_since_prior_order"].max().reset_index()

order_frequency.head()
plt.figure(figsize=(20,15))

sns.countplot(order_frequency["days_since_prior_order"])

plt.title("How often Customers Purchase")

plt.xlabel("Day Since Prior Order")

plt.ylabel("Count")
#looking at the products

products.head()
# products per order

product_amount_per_order=products.groupby("order_id")["add_to_cart_order"].max().reset_index()

product_amount_per_order.head()
product_amount_per_order["add_to_cart_order"].value_counts()
plt.figure(figsize=(20,15))

sns.countplot(product_amount_per_order["add_to_cart_order"])

plt.title("Number of Products per Order")

plt.xlabel("Number of Products")

plt.ylabel("Frequency")
# most ordered products

top_ten_products=products["product_name"].value_counts().head(10)

top_ten_products
plt.figure(figsize=(15,10))

sns.countplot(x="product_name", hue="department", data=products, order=products.product_name.value_counts().iloc[:10].index)

plt.title("Top Ten Purchased Products")

plt.xlabel("Product Name")

plt.ylabel("Count")
# look at the reorders

product_reorders=products.groupby(['product_id', 'product_name'])['reordered'].count().reset_index()

product_reorders_top_ten=product_reorders.nlargest(10, "reordered")

product_reorders_top_ten
plt.figure(figsize=(15,10))

sns.barplot(x="product_name", y="reordered", data=product_reorders_top_ten)
# top ten reorder products and aisles

product_orders=products.groupby(['product_id', 'product_name', "aisle"])[['order_id']].count().reset_index()

product_orders.columns=["product_id", "product_name", "aisle", "order_amount"]

product_orders.head()
product_orders_top_ten=product_orders.nlargest(10, 'order_amount')

product_orders_top_ten
# look at the top aisles

top_aisle_in_one_order=products.groupby("aisle")["order_id"].count().reset_index()

top_aisle_in_one_order=top_aisle_in_one_order.nlargest(10, "order_id")

top_aisle_in_one_order
plt.figure(figsize=(20,10))

sns.barplot(x="aisle", y="order_id", data=top_aisle_in_one_order)
#look at the products, customer and order mapping

products_and_orders.head()
#look at the day of order and the basket size

percentage_of_orders=products_and_orders.groupby("order_dow")["order_id"].count().reset_index()

percentage_of_orders["percentage"]=percentage_of_orders["order_id"]/percentage_of_orders["order_id"].sum()

percentage_of_orders.head()
plt.figure(figsize=(20,10))

sns.barplot(x="order_dow", y="percentage", data=percentage_of_orders)
# look at orders and products

orders_in_a_day=products_and_orders.groupby(["order_dow", "product_name"])["order_id"].count().reset_index()

orders_in_a_day["percentage"]=orders_in_a_day["order_id"]/orders_in_a_day["order_id"].sum()

orders_in_a_day=orders_in_a_day.nlargest(10, "percentage")

orders_in_a_day
plt.figure(figsize=(15,10))

sns.barplot(x="order_dow", y="percentage", data=orders_in_a_day, hue="product_name")
# Looking at departments

top_departments=products_and_orders.groupby(["department"])["order_id"].count().reset_index()

top_departments=top_departments.nlargest(10, "order_id")

top_departments
plt.figure(figsize=(15,10))

sns.barplot(x="department", y="order_id", data=top_departments)