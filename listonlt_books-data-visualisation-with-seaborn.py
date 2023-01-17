import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

df = pd.read_csv('../input/books.csv',error_bad_lines=False)

df.head(2)
df.shape #finding the number of rows and columns in the dataset
df.drop(['bookID','isbn','isbn13'],axis = 1,inplace=True) # dropping unnecessary columns

df.head(2)
df.rename(columns={'# num_pages':'total_pages'},inplace=True) #renaming columns

df.head(2)
ax =df.groupby('language_code')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('language_code').plot.bar(figsize=(14,8),rot=0)

plt.title('language code',fontsize=20)

plt.xticks(fontsize=15)

for p in ax.patches:

    ax.annotate(str(p.get_height()),(p.get_x()+0.1,p.get_height()+100))
df = df[(df.language_code =='eng')|(df.language_code =='eng-US')|(df.language_code =='eng-GB')] # selecting only English books

df.head(2)


most_rated = df.sort_values('ratings_count',ascending=False).head(10).set_index('title')

plt.subplots(figsize=(12,8))

ax = most_rated['ratings_count'].sort_values().plot.barh(width=0.9,color=sns.color_palette('CMRmap',12))

ax.set_xlabel("Total ratings count ", fontsize=15)

ax.set_ylabel("Books", fontsize=15)

ax.set_title("Top 10 most rated books",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+3, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
most_text_reviews = df.sort_values('text_reviews_count',ascending=False).head(10).set_index('title')

plt.subplots(figsize=(12,8))

ax = most_text_reviews['text_reviews_count'].sort_values().plot.barh(width=0.9,color=sns.color_palette('gnuplot2',12))

ax.set_xlabel("Total text reviews count ", fontsize=15)

ax.set_ylabel("Books", fontsize=15)

ax.set_title("Top 10 books with most text reviews",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+3, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
high_average_rating =df[df['ratings_count'] > 1000]

high_average_rating = high_average_rating.sort_values('average_rating',ascending=False).head(10).set_index('title')

plt.subplots(figsize=(12,8))

ax = high_average_rating['average_rating'].sort_values().plot.barh(width=0.9,color=sns.color_palette('Spectral',12))

ax.set_xlabel("Average rating ", fontsize=15)

ax.set_ylabel("Books", fontsize=15)

ax.set_title("Top 10 books with highest average rating",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.05, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
author_of_most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')

plt.subplots(figsize=(12,8))

ax = author_of_most_books['title'].sort_values().plot.barh(width=0.9,color=sns.color_palette('gist_rainbow',12))

ax.set_xlabel("Total books ", fontsize=15)

ax.set_ylabel("Authors", fontsize=15)

ax.set_title("Top 10 authors with most books",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
dist = df

dist.average_rating = dist.average_rating.astype(float)

fig, ax = plt.subplots(figsize=[14,8])

sns.distplot(dist['average_rating'],ax=ax)

ax.set_title('Average rating distribution for all books',fontsize=20)

ax.set_xlabel('Average rating',fontsize=13)


most_pages = df.sort_values('total_pages',ascending=False).head(10).set_index('title')

plt.subplots(figsize=(12,8))

ax = most_pages['total_pages'].sort_values().plot.barh(width=0.9,color=sns.color_palette('Spectral',12))

ax.set_xlabel("Total pages ", fontsize=15)

ax.set_ylabel("Books", fontsize=15)

ax.set_title("Top 10 books with most pages",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.05, i.get_y()+.2,str(round(i.get_width())), fontsize=20,color='black')

plt.show()
highly_rated_author =df[df['average_rating']>4.4]

highly_rated_author = highly_rated_author.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')

plt.subplots(figsize=(12,8))

ax = highly_rated_author['title'].sort_values().plot.barh(width=0.9,color=sns.color_palette('terrain',12))

ax.set_xlabel("Total books ", fontsize=15)

ax.set_ylabel("Authors", fontsize=15)

ax.set_title("Top 10 highly rated authors",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()