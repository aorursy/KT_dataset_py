import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from numpy import mean

plt.style.use('ggplot')
data = pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')

data.head()
data.columns
data.isnull().sum()
data.nunique()
data.dtypes
sb.countplot(x="type", data=data)
plt.figure(figsize=(20,20))

data['rating'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(10,10))

tv = data.loc[data['type']=="TV Show"]

movie = data.loc[data['type']=="Movie"]

tv['rating'].value_counts().plot.pie(autopct="%1.1f%%", label="TV Shows")

plt.figure(figsize=(10,10))

movie['rating'].value_counts().plot.pie(autopct="%1.1f%%", label="Movies")
sb.catplot(x="type", col="rating", kind="count", col_wrap=3, data=data)
plt.figure(figsize=(15,10))

plt.ylabel("Number of Releases")

data['country'].value_counts().nlargest(10).plot.bar()
top10 = data['country'].value_counts().nlargest(10)

plt.figure(figsize=(20,15))

df = data[data['country'].isin(top10.index)]

sb.countplot(x='country', hue='type', data=df)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Country", fontsize=20)

plt.ylabel("Number of Releases", fontsize=20)
plt.figure(figsize=(20,15))

df = data.loc[data['type'] == 'Movie']

top10 = df['country'].value_counts().nlargest(10)

df = df[df['country'].isin(top10.index)]

df['duration'] = df['duration'].str.strip('min')

df['duration'] = pd.to_numeric(df['duration'])

sb.barplot(x='country', y='duration', estimator=mean, data=df, ci=False)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Country", fontsize=20)

plt.ylabel("Average Movie Length (Min)", fontsize=20)