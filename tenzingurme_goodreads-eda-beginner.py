import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/books.csv', error_bad_lines = False)
df.index = df['bookID']
print("Dataset contains "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns")
df.head()
df.replace(to_replace="J.K. Rowling-Mary GrandPré", value='J.K. Rowling', inplace=True)
df.head()
df=df.drop('bookID', axis=1)
df['title'].nunique()
df['authors'].nunique()
df=df.drop('isbn', axis=1)

df=df.drop('isbn13', axis=1)
df.dtypes
df.count()
sns.set_context('poster')

plt.figure(figsize=(20,15))

book = df['title'].value_counts()[:20]

sns.barplot(x=book, y=book.index, palette='Set3')

plt.title('Most Occurring Books')

plt.ylabel("Books")

plt.xlabel("Number of occurances");
sns.set_context('paper')

plt.figure(figsize=(20,15))

author = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')

sns.barplot(author['title'], author.index, palette='Set3')

plt.title('Top 10 authors')

plt.ylabel('Authors')

plt.xlabel('Total number of books');
df.head(10)
plt.figure(figsize=(15,10))

rating = df[['average_rating', 'ratings_count']]

rating = rating.sort_values('average_rating')

sns.regplot(rating['ratings_count'], rating['average_rating'])

plt.title('Average_rating V/s Rating_count');
fig, ax=plt.subplots(1,2, figsize=(10,10))



sns.boxplot(y=df['average_rating'], data=df, ax=ax[0], color='g')

ax[0].set_title('Average_rating')





sns.boxplot(y=df['# num_pages'], data=df, ax=ax[1], color='g')

ax[0].set_title('Average_rating')

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(20,10))

sns.distplot(df['average_rating'], ax=ax[0], color='g')

ax[0].set_title('Average Rating')

sns.distplot(df['# num_pages'], ax=ax[1], color='r')

ax[1].set_title('Number of pages')

plt.show()
plt.figure(figsize=(10,10))

sns.kdeplot(df.average_rating, df['# num_pages'], cmap='Blues', shade=True, shade_lowest=True)

plt.show()
correlation = df[['average_rating','# num_pages','ratings_count','text_reviews_count']].corr()

sns.heatmap(correlation, annot=True, vmax=1, vmin=-1, center=0)

plt.show()
df['language_code'].unique()
freq_table_lang = pd.DataFrame(df.language_code.value_counts())

freq_table_lang
sns.set_context('poster')

plt.figure(figsize=(15,10))

sns.barplot(freq_table_lang.index, freq_table_lang['language_code'])

plt.xticks(rotation=90)

plt.title('Based on Language')

plt.ylabel('Frequency Distribution')

plt.xlabel('Languages');
freq_table_lang[:7].plot(kind='pie', subplots=True, figsize=(10,10))

plt.show()
top=df[df['average_rating']==5.0]

top[['title','authors', 'average_rating']]
plt.figure(figsize=(10,15))

top = top.sort_values('ratings_count', ascending=False).head(10)

sns.barplot(x=top['ratings_count'], y=top.title, palette='Set3');
top = df.sort_values('ratings_count', ascending=False).head(10)

plt.figure(figsize=(20,10))

sns.barplot(x='average_rating', y=top.title, data=top);
plt.figure(figsize=(20,15))

author = top.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).set_index('authors')

sns.barplot(x=author['title'],y=author.index, palette='Set3');

plt.xticks([0,1,2,3,4,5])

plt.title('authors of top books');

top=top.sort_values('text_reviews_count', ascending=False)

plt.figure(figsize=(20,20))

sns.lmplot(x='average_rating', y='text_reviews_count', data=top, palette='Set3')

plt.title("text reviews VS average rating")

plt.show()
#Stephen King

authors = ['Stephen King', 'Agatha Christie', 'Dan Brown', 'J.K. Rowling']

authors = df[df['authors']==authors[0]]

authors = authors[authors['language_code']=='eng']

plt.figure(figsize=(20,15))

sns.barplot(authors['title'], authors.average_rating, palette='Set3')

plt.xticks(rotation=90)

plt.title('Stephen King\'s Books');
#Agatha Christie

authors = ['Stephen King', 'Agatha Christie', 'Dan Brown', 'J.K. Rowling']

authors = df[df['authors']==authors[1]]

authors = authors[authors['language_code']=='eng']

plt.figure(figsize=(20,20))

sns.barplot(authors['title'], authors.average_rating, palette='Set3')

plt.xticks(rotation=90)

plt.title('Agatha Christie\'s Books');
#Dan Brown

authors = ['Stephen King', 'Agatha Christie', 'Dan Brown', 'J.K. Rowling']

authors = df[df['authors']==authors[2]]

authors = authors[authors['language_code']=='eng']

plt.figure(figsize=(20,15))

sns.barplot(authors['title'], authors.average_rating, palette='Set3')

plt.xticks(rotation=90)

plt.title('Dan Brown\'s Books');
#J.K. Rowling

authors = ['Stephen King', 'Agatha Christie', 'Dan Brown', 'J.K. Rowling']

authors = df[df['authors']==authors[3]]

authors = authors[authors['language_code']=='eng']

plt.figure(figsize=(20,15))

sns.barplot(authors['title'], authors.average_rating, palette='Set3')

plt.xticks(rotation=90)

plt.title('J.K. Rowling\'s Books');