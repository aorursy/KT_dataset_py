

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



path='/kaggle/input/goodreadsbooks/books.csv'

df=pd.read_csv(path,error_bad_lines=False)
df.head(5)
df.describe()
df.info()
df.dtypes
df['authors'].unique()
df.replace(to_replace='J.K. Rowling/Mary GrandPré',value='J.K. Rowling',inplace=True)
#print(("{0} wrotes {1} books").format(df['authors'],))

df['authors'].value_counts().head(10)
df.rename(columns={'  num_pages':'Total_page'},inplace=True)
df.drop(['bookID', 'isbn', 'isbn13'],axis=1,inplace=True)
df.columns




sns.boxplot("Total_page", data=df,

               palette=["lightblue"],sym='');





sns.violinplot("text_reviews_count", data=df,

               palette=["lightblue"]);

top_author=df['authors'].value_counts()[:15]



sns.barplot(top_author.values, top_author.index, alpha=1.).set_title('Top 15 authors in terms of publications')



plt.show()



most_occur=df['title'].value_counts()[:10]



sns.barplot(most_occur.values, most_occur.index, alpha=1.).set_title('Most occuring books')



plt.show()





language_count=df['language_code'].value_counts()[:10]



sns.barplot( language_count.index,language_count.values, alpha=.7).set_title('Most occuring books')



plt.show()



most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['ratings_count'], most_rated.index, alpha=.8,palette='Set3')
most_reviews = df.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

sns.barplot(most_rated['text_reviews_count'], most_rated.index, alpha=.8,palette='Set3')


publisher=df['publisher'].value_counts()[:10]



sns.barplot(publisher.values, publisher.index, alpha=.7).set_title('Top 10 publishers')



plt.show()





pages=df['Total_page'].value_counts()[:10]



sns.barplot(pages.values, pages.index, alpha=.7).set_title(' ')



plt.show()



rev=df['text_reviews_count']

ratings=df['ratings_count']>3

dff=pd.DataFrame(df[rev & ratings].sort_values('average_rating', ascending = True).head(5))



plt.figure(figsize=(15,10))

sns.barplot(y=dff['authors'], x=dff['average_rating'], palette='Accent')
sns.boxplot(x=df['average_rating'],sym='')
plt.figure(figsize=(10,5))



df.groupby(['average_rating','title']).Total_page.sum().nlargest(10).plot(kind='barh',color='b')

plt.figure(figsize=(10,5))



df.groupby(['authors','title','average_rating']).average_rating.sum().nlargest(10).plot(kind='barh',color='g')

plt.figure(figsize=(10,5))



df.groupby(['publisher','title']).average_rating.sum().nlargest(10).plot(kind='barh',color='g')

plt.figure(figsize=(10,5))



df.groupby(['title','language_code','average_rating']).average_rating.sum().nlargest(10).plot(kind='barh',color='r')

most_reviews=df.groupby('title')['text_reviews_count'].sum().sort_values(ascending=False).head(10).plot(kind='barh',color='khaki')





dff = df.groupby(pd.cut(df['average_rating'], [0,1,2,3,4,5]))

dff = dff[['ratings_count']]

dff.sum().T