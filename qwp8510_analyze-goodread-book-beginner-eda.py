%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
df = pd.read_csv('../input/books.csv',error_bad_lines=False)

df.head()
df = df.drop(['isbn','isbn13'],axis=1)
df.head()
df = df.rename(columns={'# num_pages':'num_pages'})
df.info()
df.describe()
df[df['num_pages']==0]['bookID']
sum_pages = sum(df['num_pages'])

num_pages = len(df['num_pages'])

avg_pages = sum_pages / num_pages

for i in range(num_pages):

    if (df['num_pages'][i] == 0):

        df['num_pages'][i] = avg_pages  
df.describe()
for i,name in enumerate(df['authors']):

    if name.split('-') != [name]:

        df['authors'][i] = name.split('-')[0]
df.head()
a = 'J.K. Rowling-Mary GrandPré'

if a.split('-') != [a]:

    print(a.split('-')[0])
num_author = len(pd.unique(df['authors']))

num_language = len(pd.unique(df['language_code']))

print('number of authors:{},  number of language:{}'.format(num_author,num_language))
df[df['average_rating']==5] = 0
print(df['average_rating'].sort_values(ascending=False).head(5))

df.sort_values(by='average_rating',ascending=False).head(5).plot.bar(x='authors',y='average_rating')

plt.title('top 5 authors with average rating')



df.sort_values(by='average_rating',ascending=False).head(5).plot.bar(x='title',y='average_rating')

plt.title('top 5 title with average rating')
df.sort_values(by='ratings_count',ascending=False).head(5).plot.bar(x='authors',y='ratings_count')

plt.title('top 5 authors with ratings count')



df.sort_values(by='ratings_count',ascending=False).head(5).plot.bar(x='title',y='ratings_count')

plt.title('top 5 title with ratings count')
df.sort_values(by='text_reviews_count',ascending=False).head(5).plot.bar(x='authors',y='text_reviews_count')

plt.title('top 5 authors with text reviews count')



df.sort_values(by='text_reviews_count',ascending=False).head(5).plot.bar(x='title',y='text_reviews_count')

plt.title('top 5 title with text reviews count')
five_text_reviews = (df['text_reviews_count'].sort_values(ascending=False).head(5)).index

for i in five_text_reviews:

    print('{}:{}'.format(df['title'][i],df['average_rating'][i]))
plt.scatter(df['ratings_count'],df['average_rating'])

plt.xlabel('ratings count')

plt.ylabel('average rating')

plt.show()
plt.scatter(df['text_reviews_count'],df['average_rating'])

plt.xlabel('text reviews count')

plt.ylabel('average rating')

plt.show()
plt.scatter(df['text_reviews_count'],df['ratings_count'])

plt.xlabel('text reviews count')

plt.ylabel('ratings_count')

plt.show()
plt.figure(figsize=(20,10))

sns.countplot('language_code',data=df,palette='bright')
avg_eng = np.average(df[df['language_code']=='eng']['num_pages'])

avg_en_us = np.average(df[df['language_code']=='en-US']['num_pages'])

avg_spa = np.average(df[df['language_code']=='spa']['num_pages'])



x = ['avg_eng','avg_en_us','avg_spa']

y = [avg_eng,avg_en_us,avg_spa]

plt.bar(x,y)

plt.xlabel('Language')

plt.ylabel('average of pages')

plt.show()
author_avg_pages_count = []

ratings_count_authors = []

ratings_count_index = df['ratings_count'].sort_values(ascending=False).head(5).index

for index in ratings_count_index:    

    author_avg_pages = np.average(df[df['authors']==df['authors'][index]]['num_pages'])

    author_avg_pages_count.append(author_avg_pages)

    ratings_count_authors.append(df['authors'][index])



print(ratings_count_authors)

print(author_avg_pages_count)

plt.figure(figsize=(10,5))

plt.bar(ratings_count_authors,author_avg_pages_count)

plt.xlabel('TOP 5 ratings count authors')

plt.ylabel('average of pages')

plt.show()
author_avg_pages_count = []

ratings_count_authors = []

ratings_count_index = df['text_reviews_count'].sort_values(ascending=False).head(5).index

for index in ratings_count_index:    

    author_avg_pages = np.average(df[df['authors']==df['authors'][index]]['num_pages'])

    author_avg_pages_count.append(author_avg_pages)

    ratings_count_authors.append(df['authors'][index])



print(ratings_count_authors)

print(author_avg_pages_count)

plt.figure(figsize=(10,5))

plt.bar(ratings_count_authors,author_avg_pages_count)

plt.xlabel('TOP 5 text reviews count authors')

plt.ylabel('average of pages')

plt.show()
avg_eng = np.average(df[df['language_code']=='eng']['ratings_count'])

avg_en_us = np.average(df[df['language_code']=='en-US']['ratings_count'])

avg_spa = np.average(df[df['language_code']=='spa']['ratings_count'])



x = ['avg_eng','avg_en_us','avg_spa']

y = [avg_eng,avg_en_us,avg_spa]

plt.bar(x,y)

plt.xlabel('Language')

plt.ylabel('average of ratings count')

plt.show()
avg_eng = np.average(df[df['language_code']=='eng']['text_reviews_count'])

avg_en_us = np.average(df[df['language_code']=='en-US']['text_reviews_count'])

avg_spa = np.average(df[df['language_code']=='spa']['text_reviews_count'])



x = ['avg_eng','avg_en_us','avg_spa']

y = [avg_eng,avg_en_us,avg_spa]

plt.bar(x,y)

plt.xlabel('Language')

plt.ylabel('average of text reviews count')

plt.show()