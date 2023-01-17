import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
os.listdir("../input/superstore-data")
data = pd.read_csv("../input/superstore-data/superstore_dataset2011-2015.csv", encoding="ISO-8859-1")
#Top 20 most profitable customers

sortedTop20 = data.sort_values(['Profit'], ascending=False).head(20)

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.barplot(x='Customer Name', y='Profit', data=sortedTop20, ax=ax)

ax.set_title("Top 20 profitable Customers")

ax.set_xticklabels(p.get_xticklabels(), rotation=75)

plt.tight_layout()

plt.show()
#What is the distribution of our customer segment

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.countplot(x="Segment", data=data, ax=ax)

ax.set_title("Customer Distribution by Segment")

ax.set_xticklabels(p.get_xticklabels(), rotation=90)

for p in ax.patches:

    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.tight_layout()

plt.show()
#3. Who are our top-20 oldest customers

oldCustomers = data.sort_values(["Order Date"], ascending=True).iloc[0:20,6]

oldCustomers
#Year-wise sales and profit

data["Order_Year"] = pd.to_datetime(data["Order Date"])

data["Year"] = data["Order_Year"].dt.year



yearwiseSalesAndProfit = data.groupby("Year").agg({"Sales":np.sum, "Profit": np.sum})

yearwiseSalesAndProfit
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(221)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Profit", data=yearwiseSalesAndProfit, palette="winter", ax=ax)

ax.set_title("Year-wise Profit")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

ax = fig.add_subplot(222)

p=sns.barplot(x=yearwiseSalesAndProfit.index,y="Sales", data=yearwiseSalesAndProfit, palette="spring", ax=ax)

ax.set_title("Year-wise Sales")

ax.set_xticklabels(p.get_xticklabels(), rotation=0)

plt.tight_layout()

plt.show()
#Relationship between sales and profit -- use scatter plot

regionwiseSalesAndProfit = data.groupby("Region").agg({"Sales":np.sum, "Profit": np.sum})

regionwiseSalesAndProfit

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.scatterplot(x="Sales", y="Profit", hue=regionwiseSalesAndProfit.index, data=regionwiseSalesAndProfit) # kind="scatter")

ax.set_title("Relationship between Sales and Profit by Region")

plt.tight_layout()

plt.show()

#Relationship between sales and profit -- using scatter plot

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p = sns.pointplot(x="Region", y="Sales", data=data, color="Red", alpha=0.8) # kind="scatter")

p2 = sns.pointplot(x="Region", y="Profit", data=data, color="Lime", alpha=0.8) # kind="scatter")

ax.set_title("Relationship between Sales and Profit by Segment")

plt.tight_layout()

plt.show()
