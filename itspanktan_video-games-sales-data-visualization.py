import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.dates import DateFormatter
os.listdir("../input/")

data = pd.read_csv('../input/vgsales.csv')
data.shape
data.head()
data.info()
plt.figure(figsize=(20, 10))

df2 = data[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum().reset_index()

df2.columns = ["Region","Total_Sales"]

plt.pie(df2.loc[:3, "Total_Sales"], labels = df2.loc[:3, "Region"])
plt.figure(figsize=(15, 10))

ax = sns.barplot("Genre", "Global_Sales",ci = None, estimator = np.sum, data = data)

ax.set_title("Global Sales By Genre")
df2 = data[["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].sum().reset_index()

df2.columns = ["Region","Total_Sales"]

ax = sns.barplot("Region","Total_Sales", data = df2)

ax.set_title("Region Wise Global Sales")
plt.figure(figsize=(20, 10))

ax = sns.barplot("Year","Global_Sales", data= data, ci= None, estimator=np.sum)

plt.xticks(rotation=90)

ax.set_title("Year Wise Global Sales")
data['decades'] = pd.cut(data.Year, 4, ["1980", "1990", "2000", "2020"])

ax = sns.barplot("decades", "Global_Sales", ci= None, data=data, estimator=np.sum)

ax.set_title("Decade Wise Global Sales")

plt.xticks(rotation=30)
plt.figure(figsize=(20, 10))

sns.barplot("decades","Global_Sales", hue="Genre", ci=None, estimator=np.sum, data=data, palette="muted")
ax = ""

plt.figure(figsize=(20, 10))

for region in ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]:

    ax = sns.lineplot(y=region, x="Year", ci = None, data = data, legend="full", estimator=np.sum)

ax.set_title("Region Wise Sales Comparison")

ax.set_xlabel("Year")

ax.set_ylabel("Sales in Millions")

ax.legend(["NA_Sales","EU_Sales","JP_Sales","Other_Sales"])
plt.figure(figsize=(20, 10))

sns.barplot("Platform","Global_Sales", ci=None, estimator=np.sum, data=data, palette="muted")
plt.figure(figsize=(20, 10))

df = data.groupby(["Year","Genre","Platform"])

sns.barplot("Year", "Global_Sales", hue="Genre", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(50).reset_index(),  palette="muted")
plt.figure(figsize=(20, 10))

df = data.groupby(["Publisher"])

sns.barplot("Global_Sales", "Publisher", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(20).reset_index(),  palette="muted")
plt.figure(figsize=(20, 10))

df = data.groupby(["Name"])

sns.barplot("Global_Sales", "Name", ci= None, data =df.Global_Sales.sum().sort_values(ascending=False).head(20).reset_index(),  palette="muted")