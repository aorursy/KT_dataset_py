# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False, index_col = 'bookID')
df.head()
df.shape
df.info()
plt.figure(figsize=(16,10))
plt.title('Languages in the dataset', fontsize = 14)
sns.countplot(df['language_code'])
plt.xticks(fontsize = 10)
plt.show()
lang = df.loc[df.language_code.isin(['eng', 'en-US', 'spa', 'en-GB', 'fre', 'ger', 'jpn'])]

plt.figure(figsize=(16,10))
plt.title('Languages in the dataset', fontsize = 14)
graph = sns.countplot(lang['language_code'])
plt.xticks(fontsize = 12)
plt.xlabel('Languages', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/4., height + 0.1, height, fontsize = 14)
plt.show()
most_books = df['authors'].value_counts()[:15]
plt.figure(figsize = (14,6))
sns.barplot(x = most_books, y = most_books.index, palette = 'Blues_d')
plt.title('Authors with highest number of books')
plt.xlabel('Number of books', fontsize = 12)
plt.show()
most_ratings = df[['ratings_count']].set_index(df['title']).sort_values(by = 'ratings_count', ascending = False)[:15]
plt.figure(figsize = (12,6))
sns.barplot(x = most_ratings['ratings_count'], y = most_ratings.index, palette = 'rocket')
plt.yticks(fontsize = 10)
plt.xlabel('Number of ratings', fontsize = 12)
plt.title('The books with highest number of ratings', fontsize = 16)
plt.show()
most_reviews = df[['text_reviews_count']].set_index(df['title']).sort_values(by = 'text_reviews_count', ascending = False)[:15]
plt.figure(figsize = (12,6))
sns.barplot(x = most_reviews['text_reviews_count'], y = most_reviews.index, palette = 'Greens_d')
plt.yticks(fontsize = 10)
plt.xlabel('Review Counts', fontsize = 12)
plt.title('The books with highest number of reviews', fontsize = 16)
plt.show()
publisher = df['publisher'].value_counts().head(15)
plt.figure(figsize = (14,6))
graph = sns.barplot(y = publisher, x = publisher.index)
plt.title('Publisher with highest number of books', fontsize = 14)
plt.xlabel('Publisher', fontsize = 12)
plt.ylabel('Count')
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/4., height + 0.9, height)
plt.xticks(rotation = 45, fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
plt.figure(figsize = (14,6))
sns.distplot(df['average_rating'], bins = 30)
plt.title('Average ratings of books', fontsize = 16)
plt.xticks(fontsize = 14)
plt.xlabel('Average Ratings', fontsize = 12)
plt.show()
df.columns = df.columns.str.replace(' ', '')
plt.figure(figsize = (14,8))
sns.distplot(df['num_pages'], bins = 50)
plt.title('Average number of pages', fontsize = 14)
plt.xlabel('Number of Pages', fontsize = 12)
plt.show()
a = df.loc[(df.num_pages < 2000)]
plt.figure(figsize = (14,8))
sns.distplot(a['num_pages'], bins = 60)
plt.title('Average number of pages', fontsize = 14)
plt.xlabel('Number of pages', fontsize = 12)
plt.show()
plt.figure(figsize=(14,6))
df.dropna(0, inplace=True)
sns.scatterplot(x = 'average_rating', y = 'text_reviews_count', data = df, color = 'red')
plt.title('Ratings vs Review counts', fontsize = 14)
plt.xlabel('Average ratings', fontsize = 12)
plt.ylabel('Reviews count', fontsize = 12)
plt.show()
a = df.loc[(df.text_reviews_count < 4000)]
plt.figure(figsize=(16,10))
df.dropna(0, inplace=True)
sns.jointplot(x = 'average_rating', y='text_reviews_count', data = a, color = 'red')
plt.show()
plt.figure(figsize=(16,10))
sns.scatterplot(x = 'average_rating', y = 'num_pages', data = df, color = 'g')
plt.title('Ratings vs Number of Pages', fontsize = 14)
plt.xticks(fontsize = 12)
plt.ylabel('Number of pages', fontsize = 12)
plt.xlabel('Average Ratings', fontsize = 12)
plt.show()
a = df.loc[(df.num_pages < 1000)]
plt.figure(figsize=(16,10))
sns.jointplot(x = 'average_rating', y = 'num_pages', data = a, color = 'darkgreen')
plt.show()
