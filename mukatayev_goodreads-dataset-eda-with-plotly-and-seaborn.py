# Importing required libraries

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px
pd.set_option('display.max_rows', 10000)

pd.set_option('display.max_columns', 1000)
df = pd.read_csv('../input/books.csv', error_bad_lines=False)
df.head()


print(df.shape)
# check missing values for each column 

df.isnull().sum().sort_values(ascending=False)
# check out the rows with missing values

df[df.isnull().any(axis=1)].head()
print(df.info())
df.describe()
df.describe(include=['object'])
print(df.columns)
#Rearrange the columns to easier reference

df = df[['bookID', 'title', 'authors', 'average_rating', 

       'language_code', '# num_pages', 'ratings_count', 'text_reviews_count', 'isbn', 'isbn13']]
df.head()
sorted_rated_df = df.sort_values(by='ratings_count', ascending=False)[:10]



fig = plt.figure(figsize=(18,10))



sns.barplot(x=sorted_rated_df['ratings_count'], y=sorted_rated_df['title'], palette="rainbow")

plt.title('The top 10 popular books')

plt.show()
plt.figure(figsize=(18,10))

titles = df['title'].value_counts()[:10]

sns.barplot(x = titles, y = titles.index, palette='rainbow')

plt.title("10 most Occurring Books")

plt.xlabel("Number of occurrences")

plt.ylabel("Books")

plt.show()
corr_columns = ['average_rating', 'ratings_count', 'text_reviews_count']

corr_mat = df[corr_columns].corr()

corr_mat
#Plotting the Correlation matix

plt.figure(figsize=(14,14))

corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(10, 650, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=55,

    horizontalalignment='right'

);
#Sort all languages that appear less than 200 times (< 200) into category "other" because their number are too small

language_count_df = df[['language_code']].copy()

language_count_df.head()

language_count_df.loc[language_count_df['language_code'].isin((language_count_df['language_code'].value_counts()[language_count_df['language_code'].value_counts() < 200]).index), 'language_code'] = 'other'
import plotly.graph_objects as go

labels=language_count_df.language_code.unique()

values=language_count_df['language_code'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
fig = px.scatter(df,x="average_rating", y="# num_pages")

fig.show()
fig = px.scatter(df,x="ratings_count", y="# num_pages")

fig.show()
fig = px.scatter(df,y="ratings_count", x="text_reviews_count")

fig.show()
#Sorting the dataset by authors with highest number of books

sorted_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False)[:10]

sorted_books.head()
fig = plt.figure(figsize=(18,10))



sns.barplot(y=sorted_books['title'], x=sorted_books['authors'], palette="Set1")

plt.title('The top 10 authors with most books')

plt.show()
sorted_authors_ = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False)
high_rated_author = df[df['average_rating']>=4.3]

high_rated_author = high_rated_author.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')

plt.figure(figsize=(15,10))

ax = sns.barplot(high_rated_author['title'], high_rated_author.index, palette='Set1')

ax.set_xlabel("Number of Books")

ax.set_ylabel("Authors")

for i in ax.patches:

    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')