import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import os
path="../input"
os.chdir(path)
store=pd.read_csv("superstore_dataset2011-2015.csv",encoding="ISO-8859-1")
store.info()
store.shape[0] # How many rows
store.shape[1] # How many columns
store.index.values # Get the row names
store.columns.values # Get the columns names
store["Product Name"]
y=store.sort_values("Profit", ascending=False).iloc[0:20,:]
y ["Customer Name"] # Who are the top 20 most profitable customers
store["Segment"].unique() # What is the distribution of our customer segment
z=store.sort_values(["Order Date"],ascending=True).iloc[0:20,:]
z["Customer Name"].unique() # Who are the top 20 oldest customers
store["Customer ID"].unique()
store.sort_values(["Profit","Discount"],ascending=[True,False])
plt.xticks(rotation=90),sns.countplot("Sub-Category", data = store)
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot("Category", data = store)
sns.countplot("Category",hue="Sub-Category" ,data = store)
plt.xticks(rotation=90),sns.barplot(x="Customer Name",y="Profit",data=store) # plot of top 20 most profitable customers 
sns.barplot(x="Category",y="Profit",hue="Sub-Category",data=store)
sns.countplot("Market",data=store)
sns.boxplot("Order Priority","Profit",data=store)
store.groupby("Customer ID").apply(lambda x:pd.Series(dict(store_visit=x.shape[0]))).reset_index()