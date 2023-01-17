import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from tqdm import tqdm

from progressbar import ProgressBar

import re

from scipy.cluster.vq import kmeans, vq

from pylab import plot, show

from matplotlib.lines import Line2D

import matplotlib.colors as mcolors

from sklearn.cluster import KMeans

from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../input/books.csv', error_bad_lines = False)

print('Size of weather data frame is :',df.shape)
df.dtypes
#Converting object datatypes to numeric

df[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']] = df[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']].convert_objects(convert_numeric=True)

df['isbn13'] = df['isbn13'].apply(str)

df.dtypes
#How many null values do we have?

df.count().sort_values()
plt.figure(figsize=(10,10))

plot = sns.countplot(y = "authors", data = df, order = df['authors'].value_counts().iloc[:10].index, palette = "Set3")
most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['ratings_count'], most_rated.index, palette='Set3')
plt.figure(figsize=(10,10))

plot = sns.countplot(y = "average_rating", data = df, order = df['average_rating'].value_counts().iloc[:10].index, palette = "Set3")
sns.distplot(df['average_rating'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});
sns.distplot(df['# num_pages'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});
most_rated = df.sort_values('# num_pages', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['# num_pages'], most_rated.index, palette='Set3')
sns.distplot(df['text_reviews_count'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});
most_rated = df.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['text_reviews_count'], most_rated.index, palette='Set3')
plt.figure(figsize=(15,10))

df.dropna(0, inplace=True)

sns.set_context('paper')

ax =sns.jointplot(x="average_rating",y='text_reviews_count', kind='scatter',  data= df[['text_reviews_count', 'average_rating']])

ax.set_axis_labels("Average Rating", "Text Review Count")

plt.show()
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="average_rating", y="# num_pages", data = df, color = 'crimson')

ax.set_axis_labels("Average Rating", "Number of Pages")
without_pages_outliers = df[~(df['# num_pages']>1500)]

ax = sns.jointplot(x="average_rating", y="# num_pages", data = without_pages_outliers, color = 'darkcyan')

ax.set_axis_labels("Average Rating", "Number of Pages")