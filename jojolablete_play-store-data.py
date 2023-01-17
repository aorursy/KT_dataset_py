# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')

reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')

data.head()
reviews.head()
data.info()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize = (10, 4))

ax = sns.countplot(data['Category'], orient = 'v')

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.show()
plt.figure()

plt.title('Do we have more free apps than pay apps?')

ax = sns.countplot(data['Type'].loc[data['Type'] != '0'])

plt.show()
data['Price'] = data['Price'].astype(dtype = 'str')

data['Price'].unique()
data['Price'] = data['Price'].str.replace('$', '', regex = False)

data.drop(data.index[data['Price'] == 'Everyone'], axis = 0, inplace = True) # Check if everyone = Free

data['Price'] = data['Price'].astype(dtype = 'float64')

sns.distplot(data['Rating'].loc[data['Rating'].notna()])
plt.figure()

sns.scatterplot(x = 'Rating', y = 'Reviews', data = data)

plt.plot()
rating_cat = data.groupby('Category')['Rating'].mean().reset_index()



plt.figure(figsize = (12,4))

ax = sns.barplot(x = 'Category', y = 'Rating', data = rating_cat)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.plot()
data['Reviews'] = data['Reviews'].astype(dtype = 'int64')

ax = sns.distplot(data['Reviews'])
reviews_cat = data.groupby('Category')['Reviews'].mean().reset_index()

plt.figure(figsize = (12, 4))

ax = sns.barplot(x = 'Category', y = 'Reviews', data = reviews_cat)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.plot()
plt.figure()

sns.scatterplot(x = 'Type', y = 'Reviews', data = data)

plt.plot()
full = data.merge(reviews, how = 'inner', on = 'App')

full.head()
sentiment_app = full.groupby('App')['Sentiment_Polarity', 'Sentiment_Subjectivity'].mean().reset_index()

sentiment_app.head()
sns.distplot(sentiment_app['Sentiment_Subjectivity'].loc[sentiment_app['Sentiment_Subjectivity'].notna()])
sns.distplot(sentiment_app['Sentiment_Polarity'].loc[sentiment_app['Sentiment_Polarity'].notna()])
sentiment_app.columns = ['App', 'Mean_Sentiment_Polarity', 'Mean_Sentiment_Subjectivity']

full = full.merge(sentiment_app, on = 'App')

full.head()
sentiment_app = sentiment_app.merge(data.loc[:, ['App', 'Reviews']], on = 'App', how = 'inner')

sentiment_app.drop_duplicates(inplace = True)

sentiment_app = pd.melt(sentiment_app, id_vars = ['App', 'Reviews'], var_name = 'Sentiment')

sentiment_app.head()
sns.scatterplot(x = 'value', y = 'Reviews', hue = 'Sentiment', data = sentiment_app)
sentiment_cat = full.groupby('Category')['Sentiment_Polarity', 'Sentiment_Subjectivity'].mean().reset_index()
sentiment_cat = pd.melt(sentiment_cat, id_vars = ['Category'], var_name = 'Sentiment')

sentiment_cat.head()
plt.figure(figsize = (14, 6))

ax = sns.barplot(x = 'Category', y = 'value', hue = 'Sentiment', data = sentiment_cat)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

ax.plot()
threshold = 100000

data['killer'] = 0

data.loc[data['Reviews'] > threshold, 'killer'] = 1

data.head()