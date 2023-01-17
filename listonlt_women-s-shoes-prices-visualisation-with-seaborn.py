import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



df = pd.read_csv('../input/7210_1.csv')

df.head(2)
df.shape
df.isnull().sum()*100/df.shape[0]

df = df[['name','brand','categories','prices.amountMin','prices.amountMax','prices.currency','prices.isSale']]

df.head(2)

df.dropna(inplace=True)
df.shape
top_brands =df.groupby('brand')['name'].count().reset_index().sort_values('name',ascending=False).head(10).set_index('brand')

plt.subplots(figsize=(14,7))

ax = top_brands['name'].sort_values().plot.barh(width=0.9,color=sns.color_palette('CMRmap',12))

ax.set_xlabel("Total items", fontsize=18)

ax.set_ylabel("Brand", fontsize=18)

ax.set_title("Top 10 Brands",fontsize=18,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+.3,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
sale =df.groupby('prices.isSale')['name'].count().reset_index().sort_values('name',ascending=False).head(10).set_index('prices.isSale')

plt.subplots(figsize=(14,7))

ax = sale['name'].sort_values().plot.barh(width=0.9,color=sns.color_palette('Greens',2))

ax.set_xlabel("Total items", fontsize=18)

ax.set_ylabel("Is on sale", fontsize=18)

ax.set_title("On Sale ?",fontsize=18,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+.3,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
top_categories =df.groupby('categories')['name'].count().reset_index().sort_values('name',ascending=False).head(10).set_index('categories')

plt.subplots(figsize=(14,7))

ax = top_categories['name'].sort_values().plot.barh(width=0.9,color=sns.color_palette('coolwarm',12))

ax.set_xlabel("Total items", fontsize=18)

ax.set_ylabel("Category", fontsize=18)

ax.set_title("Top categories",fontsize=18,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+.3,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
currency =df.groupby('prices.currency')['name'].count().reset_index().sort_values('name',ascending=False).head(10).set_index('prices.currency')

plt.subplots(figsize=(14,7))

ax = currency['name'].sort_values().plot.barh(width=0.9,color=sns.color_palette('Blues',2))

ax.set_xlabel("Product Count", fontsize=18)

ax.set_ylabel("Currency", fontsize=18)

ax.set_title("Currency Type",fontsize=18,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+.3,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
df =df.sort_values('prices.amountMax',ascending=False)

costliest_products =df.groupby('brand')['prices.amountMax'].first().reset_index().sort_values('prices.amountMax',ascending=False).head(10).set_index('brand')

plt.subplots(figsize=(14,7))

ax = costliest_products['prices.amountMax'].sort_values().plot.barh(width=0.9,color=sns.color_palette('gnuplot2_r',10))

ax.set_xlabel("Cost", fontsize=18)

ax.set_ylabel("Brand", fontsize=18)

ax.set_title("Costliest products of top 10 brands",fontsize=18,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+.3,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
dist = df

dist['prices.amountMax'] = dist['prices.amountMax'].astype(float)

fig, ax = plt.subplots(figsize=[14,8])

sns.distplot(dist['prices.amountMax'],ax=ax)

ax.set_title('Average price of all shoes',fontsize=20)

ax.set_xlabel('Average price',fontsize=13)