# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head()
data.isnull().sum()
sns.heatmap(data.isnull(), annot = True)
data["country"].replace(np.nan, "United States", inplace=True)
data.head()
data.drop(['director', 'cast'], axis=1, inplace=True)
data.head()
data.drop(['date_added'], axis=1, inplace=True)
data.head()
data['type'].value_counts()
data['listed_in'].value_counts()
data['rating'].value_counts()
data.isnull().sum()
data['rating'].replace(np.nan, 'TV-MA',inplace  = True)
data.isnull().sum()
data.head()
sns.countplot(x='type', data=data)
plt.title("Type of the Show on Netflix")
plt.figure(figsize=(10, 10))
sns.countplot(x='rating',data = data)
plt.title("Rating of the show on Netflix")
plt.figure(figsize = (35,10))
sns.countplot(x='release_year',data = data)
plt.title("Release Year of the Show on Netflix")
plt.figure(figsize=(10,8))
sns.scatterplot(x='rating',y='type',data = data)
plt.title("Rating with type of show on Netflix")
plt.figure(figsize = (14,10))
sns.countplot(x='rating',data = data,hue='type')
plt.title("Rating with both the title i.e Movie and TV Show")
plt.figure(figsize=(12,6))
data[data["type"]=="Movie"]["release_year"].value_counts()[:20].plot(kind="bar",color="green")
plt.title("Frequency of Movies which were released in different years and are available on Netflix")
plt.figure(figsize=(12,6))
data[data["type"]=="TV Show"]["release_year"].value_counts()[:20].plot(kind="bar",color="yellow")
plt.title("Frequency of TV shows which were released in different years and are available on Netflix")
plt.figure(figsize=(12,6))
data[data["type"]=="Movie"]["listed_in"].value_counts()[:10].plot(kind="barh",color="pink")
plt.title("Top 10 Genres of Movies",size=18)
plt.figure(figsize=(12,6))
data[data["type"]=="TV Show"]["listed_in"].value_counts()[:10].plot(kind="barh",color="brown")
plt.title("Top 10 Genres of TV Shows",size=18)