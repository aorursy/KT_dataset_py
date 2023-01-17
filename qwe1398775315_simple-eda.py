# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
beers = pd.read_csv("../input/beers.csv")
breweries = pd.read_csv("../input/breweries.csv")
beers = beers.drop('Unnamed: 0',axis=1)
beers.head(5)
breweries = breweries.rename(columns={'Unnamed: 0':'brewery_id'},index=str)
breweries.head(5)
data = pd.merge(beers,breweries,how='left', on='brewery_id')
data.head(5)
sns.distplot(data.abv.dropna(),bins=10)
plt.xlabel('abv')
plt.ylabel('frequency')
plt.title('Distribution of abv')
sns.distplot(data['ibu'].dropna())
plt.xlabel('ibu')
plt.ylabel('frequency')
plt.title('Distribution of ibu')
data.name_x.describe()
data['style'].describe()
plt.figure(figsize=(14,28))
sns.countplot(data=data,y='style')
sns.countplot(data=data,y='ounces')
data.name_y.describe()
data.city.describe()
data.state.describe()
plt.figure(figsize=(14,14))
sns.countplot(data=data,y='state')
data[['city','style']].groupby(['city']).count().sort_values('style').reset_index().loc[::-1].head(10)
data[['state','style']].groupby(['state']).count().sort_values('style').reset_index().loc[::-1].head(10)
data[['city','abv']].dropna().groupby(['city']).mean().reset_index().sort_values('abv').loc[::-1].head(10)
data[['state','abv']].dropna().groupby(['state']).mean().reset_index().sort_values('abv').loc[::-1].head(10)
from wordcloud import WordCloud,STOPWORDS
cloud = WordCloud(stopwords=STOPWORDS)
word_cloud = cloud.generate(' '.join(data['name_x']))
plt.figure(figsize=(16,16))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
