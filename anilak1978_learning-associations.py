# import neccessary libraries

import pandas  as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#load and read data

df=pd.read_csv("https://raw.githubusercontent.com/anilak1978/ecommerce/master/OnlineRetail.csv", encoding="ISO-8859-1")

df.head()
#look at data types

df.dtypes
# Look at missing values

df.isnull().sum()
# Update Date Data Type and Clean Spaces in Description

df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])

df["InvoiceDay"]=pd.DatetimeIndex(df["InvoiceDate"]).day

df["InvoiceMonth"]=pd.DatetimeIndex(df["InvoiceDate"]).month

df["InvoiceYear"]=pd.DatetimeIndex(df["InvoiceDate"]).year

df["Hour"]=pd.DatetimeIndex(df["InvoiceDate"]).hour

df["Description"]=df["Description"].str.strip()

df.head()
df.info()
# Look for any negative values in Unit Price

print(df[df["UnitPrice"]<0])
# look for invoices with credit rather than purchase

print(df[df["InvoiceNo"].str.contains('C')])
# Clean up the non purchase invoices.

df["InvoiceNo"]=df["InvoiceNo"].astype('str')

df[df["InvoiceNo"].str.contains('C', na=False)].head()
# Remove all the returns from the dataset

df = df[~df['InvoiceNo'].str.contains('C')]

df.head()
# Look at hour of the purchases

plt.figure(figsize=(15,10))

sns.countplot(x="Hour", data=df)

plt.title("What time consumers make the purchase?")

plt.xlabel("Hour")

plt.ylabel("Count")
# Look at day of the purchases

plt.figure(figsize=(15,10))

sns.countplot(x="InvoiceDay", data=df)

plt.title("What day consumers make the purchase?")

plt.xlabel("Day")

plt.ylabel("Count")
# Look at items purchase

df_item=df.groupby(["StockCode", "Description"])["Quantity"].sum().reset_index()

df_item=df_item.nlargest(10, "Quantity")

plt.figure(figsize=(15,10))

sns.barplot(x="Quantity", y="Description", data=df_item)

plt.title("Top Ten Items Purchased")

plt.xlabel("Quantity")

plt.ylabel("Item")
# Look at the quantity of purchase for each transaction

df_invoice=df.groupby(["InvoiceNo"])["Quantity"].sum().reset_index()

df_invoice=df_invoice.nlargest(10, "Quantity")

plt.figure(figsize=(15,10))

sns.barplot(x="InvoiceNo", y="Quantity", data=df_invoice)

plt.title("Top Ten Transactions")

plt.xlabel("Quantity")

plt.ylabel("Item")
df_transactional=df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")

df_transactional.head()
# create function to encode 

def hot_encode(x):

    if(x<=0):

        return 0

    if(x>=1):

        return 1
df_transactional_encoded=df_transactional.applymap(hot_encode)

df_transactional_encoded.head()
print(df_transactional_encoded[df_transactional_encoded["10 COLOUR SPACEBOY PEN"]==1])
df_transactional_list=df.groupby(["InvoiceNo"])["Description"].apply(list)

df_transactional_list[1:10]
from mlxtend.frequent_patterns import apriori, association_rules
# look for frequent items that has 8% support

frequent_items=apriori(df_transactional_encoded, min_support=0.03, use_colnames=True)

frequent_items.head()
# look at the rules

rules = association_rules(frequent_items, metric ="lift", min_threshold = 0.05) 

rules.sort_values(["confidence", "lift"], ascending=[False, False])

rules.head()
top_frequent_items=frequent_items.nlargest(10, "support")

plt.figure(figsize=(15,10))

sns.barplot(x="itemsets", y="support", data=top_frequent_items, orient="v")

plt.title("Top 10 Frequent Items")

plt.xlabel("Support")

plt.ylabel("Itemsets")
top_frequent_items
# Look at the correlation between support and confidence

plt.figure(figsize=(15,10))

sns.scatterplot(x="support", y="confidence", data=rules)
# top ten association rules

top_ten_association_rules= rules.nlargest(10, "confidence")

top_ten_association_rules