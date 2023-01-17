import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/ecological-footprint/EcologicalFootPrint.csv")

print(df.isnull().sum())

df.dropna(inplace=True)

df = df[df['country']!='World']
df.head()
df['record'].value_counts()
l = ['crop_land','grazing_land','forest_land','fishing_ground','built_up_land','carbon','total']

for columns in l:

    plt.figure(figsize=(15,10))

    every_year = df.groupby('year')[columns].mean()

    sns.barplot(every_year.index,every_year.values).set_xticklabels(sns.barplot(every_year.index,every_year.values).get_xticklabels(),rotation="90")

    plt.title("Year Comparation with "+columns)

    plt.xlabel(columns)

    plt.ylabel("Count")
plt.figure(figsize=(10,10))

plt.title("Total 10, most Carbon Producing Countring")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['carbon'].sort_values(ascending=False)[:10].plot.bar()
plt.figure(figsize=(10,10))

plt.title("Total 10, lowest Carbon Producing Countring")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['carbon'].sort_values(ascending=True)[:10].plot.bar()
plt.figure(figsize=(10,10))

plt.title("Total 10, Lowest Country with Forst Land")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['forest_land'].sort_values(ascending=True)[:10].plot.bar()
plt.figure(figsize=(10,10))

plt.title("Total 10, Highest Country with Forst Land")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['forest_land'].sort_values(ascending=False)[:10].plot.bar()
plt.figure(figsize=(10,10))

plt.title("Total 10, Highest Country with Build Up Land")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['built_up_land'].sort_values(ascending=False)[:10].plot.bar()
plt.figure(figsize=(10,10))

plt.title("Total 10, Lowest Country with Build Up Land")

plt.xlabel("Country")

plt.ylabel("Rank")

df.groupby(['country']).mean()['built_up_land'].sort_values(ascending=True)[:10].plot.bar()