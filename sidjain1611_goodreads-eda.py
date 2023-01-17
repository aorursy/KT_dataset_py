# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/books.csv', error_bad_lines = False)
df.head()
df.describe(include='all')
df.info()
df.replace(to_replace='J.K. Rowling-Mary GrandPré', value = 'J.K. Rowling', inplace=True)
df['authors'].nunique()
plt.figure(1, figsize=(15, 7))

plt.title("Which aurthor wrote maximum books")

sns.countplot(x = "authors", order=df['authors'].value_counts().index[0:10] ,data=df)
plt.figure(1, figsize=(25,7))

plt.title("Most Occuring Books")

sns.countplot(x = "title", order=df['title'].value_counts().index[0:10] ,data=df)
plt.figure(1, figsize=(25,10))

plt.title("language_codes")

sns.countplot(x = "language_code", order=df['language_code'].value_counts().index[0:10] ,data=df)
most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['ratings_count'], most_rated.index, palette='Accent')
most_rated['num_pages']=most_rated['# num_pages']

most_rated=most_rated.drop('# num_pages',axis=1)
plt.figure(figsize=(20,8))

sns.barplot(most_rated['num_pages'], most_rated['ratings_count'], palette='Accent')
plt.figure(figsize=(10,5))

most_rated.groupby(['average_rating','title']).num_pages.sum().nlargest(10).plot(kind='barh',color='b')
df_pages=df['# num_pages'] > 5

df_rating=df['average_rating'] > 4.5

req=pd.DataFrame(df[df_pages & df_rating].sort_values('# num_pages', ascending = True).head(5))

req.head()
plt.figure(figsize=(15,10))

sns.barplot(req['# num_pages'], req['title'], palette='Accent')
most_rated = df.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['text_reviews_count'], most_rated.index, palette='Accent')
req = df.groupby(pd.cut(df['average_rating'], [0,1,2,3,4,5]))

req = req[['ratings_count']]

req.sum().reset_index()