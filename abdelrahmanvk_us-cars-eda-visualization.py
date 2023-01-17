import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

df
df.describe()
median_price = df['price'].median()

df['price'] = df['price'].astype(int)

df['price'].replace(0,median_price ,inplace=True)
df.columns

#Columns that we have in data.
#missing data

df.isnull().sum().sort_values(ascending=False)
fig, ax = plt.subplots()

sns.set_palette("ocean")

sns.countplot(x = "title_status", data = df, ax = ax)

ax.set_xlabel("Vehicle Status")

ax.set_ylabel("Count")

plt.show()
sns.swarmplot(x="price", y="title_status" ,data=df);

plt.show()
custom_palette = ["blue", "green", "orange","red","yellow", "purple"]

sns.set_palette(custom_palette)

fig, ax = plt.subplots(figsize = (16,11))

sns.scatterplot(x = "year", y = "price", hue = "title_status", data = df, ax = ax)

ax.set_xlabel("Vehicle Registration Year")

ax.set_ylabel("Sale Price($)")

plt.show()
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, figsize=(16,9))

sns.set_style("darkgrid")

sns.scatterplot(x = "price", y = "mileage", data = df, ax =ax0)

sns.scatterplot(x = "year", y = "mileage", data = df, ax= ax1)

ax0.set_xlabel("Sale Price($)")

ax1.set_xlabel("Vehicle Registration Year")

ax0.set_ylabel("Distance Travelled (Miles)")

ax1.set_ylabel("Distance Travelled (Miles)")

plt.show()
plt.figure(figsize=(12,6))

sorted_nb = df.groupby(['brand'])['price'].median().sort_values()

sns.boxplot(x=df['brand'], y=df['price'], order=list(sorted_nb.index))

plt.xticks(rotation=70)
top5_colors = list(df.color.value_counts()[0:5].index)

top5_colors
df_top5_color = df[df["color"].isin(top5_colors)]

fig, ax = plt.subplots(figsize = (16,11))

sns.boxplot(x = "color", y = "price",data = df_top5_color,palette = "inferno", ax = ax)

ax.set_xlabel("Color")

ax.set_ylabel("Sale Price($)")

plt.show()
df['state'].value_counts().head(10).plot(kind='barh', figsize=(6,10))
over100_states = df["state"].value_counts()[df["state"].value_counts() >= 100].index

df_over100_cars_per_state = df[df["state"].isin(over100_states)]

fig, ax = plt.subplots(figsize = (16,16))

sns.lvplot(data=df_over100_cars_per_state, y="state",

x="price", ax=ax)

ax.set_xlabel("Sale Price($)")

ax.set_ylabel("State")