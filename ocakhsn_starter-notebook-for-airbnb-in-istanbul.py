import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import FastMarkerCluster

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/istanbul-airbnb-dataset/listings.csv', index_col="id")

df.head()
print("There are {} rows and {} columns".format(df.shape[0], df.shape[1]))
df.isnull().sum()
df.describe()
df.drop(columns=['neighbourhood_group', 'last_review', 'reviews_per_month'], inplace=True)
plt.figure(figsize=(6,6))

sns.boxplot(y=df['price'])

plt.title("Distribution of Price")

plt.show()
mean = df['price'].mean()

std = df['price'].std()

upper_limit = mean + 3 * std

df = df[df['price'] < upper_limit]


plt.figure(figsize=(6,6))

sns.boxplot(y=df['price'] )#Show below the upper limit

plt.title("Distribution of Price")

plt.show()
plt.figure(figsize=(6,6))

numbers = df['neighbourhood'].value_counts()

plt.pie(numbers.values, labels=numbers.index, autopct='%1.1f%%')

plt.title('Numbers in Each Neigbourhoods')
plt.figure(figsize=(20, 10))

sns.boxplot(x="neighbourhood", y="price", data=df)

plt.xticks(rotation=90)

plt.show()
prices_by_neighbourhhods = df.groupby("neighbourhood")['price'].agg(['min', 'max', 'mean', 'count']).reset_index()

prices_by_neighbourhhods.sort_values(by="mean", ascending=False)
plt.figure(figsize=(6,6))

numbers = df['room_type'].value_counts()

plt.pie(numbers.values, labels=numbers.index, colors=['cyan', 'green', 'pink'], autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Numbers in Each Room Types')
df.groupby('neighbourhood')['room_type'].count()
fig = plt.figure(figsize=(15,6))



sns.scatterplot(df['longitude'], df['latitude'], hue=df['room_type'])

plt.title('Distribution of Room Types in the Map')



plt.show()