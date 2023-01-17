import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv',error_bad_lines = False)
df.head() #checking the head of our data
df.isnull().sum() #checking for any null values present in the data
df.dtypes #checking the data types of each column
df.describe() #checking for hidden values such as the maximum rating of our books, the average number of ratings
top_ten = df[df['ratings_count'] > 1000000]

top_ten.sort_values(by='average_rating', ascending=False).head(10)
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(10, 10))

data = top_ten.sort_values(by='average_rating', ascending=False).head(10)

sns.barplot(x="average_rating", y="title", data=data, palette='inferno')
most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')

plt.figure(figsize=(15,10))

ax = sns.barplot(most_books['title'], most_books.index, palette='inferno')

ax.set_title("Top 10 authors with most books")

ax.set_xlabel("Total number of books")

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
#df.head()
most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')

plt.figure(figsize=(15,10))

ax = sns.barplot(most_rated['ratings_count'], most_rated.index, palette = 'inferno')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
plt.figure(1, figsize=(25,10))

plt.title("Languages")

sns.countplot(x = "language_code", order=df['language_code'].value_counts().index[0:10] ,data=df,palette='inferno')
highly_rated_author =df[df['average_rating']>4.4]

highly_rated_author = highly_rated_author.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')

plt.subplots(figsize=(15,10))

ax = highly_rated_author['title'].sort_values().plot.barh(width=0.9,color=sns.color_palette('inferno',12))

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
top_publishers = df.groupby('publisher')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('publisher')

plt.subplots(figsize=(15,10))

ax = top_publishers['title'].sort_values().plot.barh(width=0.9,color=sns.color_palette('inferno',12))

ax.set_xlabel("Total books ", fontsize=15)

ax.set_ylabel("Publishers", fontsize=15)

ax.set_title("Top 10 Publishers Present in our data",fontsize=20,color='black')

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+.2,str(round(i.get_width())), fontsize=15,color='black')

plt.show()
df.average_rating = df.average_rating.astype(float)

fig, ax = plt.subplots(figsize=[15,10])

sns.distplot(df['average_rating'],ax=ax)

ax.set_title('Average rating distribution for all books',fontsize=20)

ax.set_xlabel('Average rating',fontsize=13)
df.info()
ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color = 'red', sizes=(100, 200), height=7, marker='o')

plt.title("Relation between Rating counts and Average Ratings",fontsize = 15)

ax.set_axis_labels("Average Rating", "Ratings Count")
plt.figure(figsize=(15,10))

ax = sns.relplot(x="average_rating", y="  num_pages", data = df, color = 'red',sizes=(100, 200), height=7, marker='o')

ax.set_axis_labels("Average Rating", "Number of Pages")
plt.figure(figsize=(15,10))

ax = sns.relplot(x="average_rating", y="language_code", data = df, color = 'red',sizes=(100, 200), height=7, marker='o')

ax.set_axis_labels("Average Rating", "Languages")
plt.figure(figsize=(15,10))

ax = sns.relplot(x="average_rating", y="text_reviews_count", data = df, color = 'red',sizes=(100, 200), height=7, marker='o')

ax.set_axis_labels("Average Rating", "Text Reviews Count")
df2 = df.copy()
df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"

df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"

df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"

df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"

df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"
df2.head()
rating_df = pd.get_dummies(df2['rating_between'])

rating_df.head()
language_df = pd.get_dummies(df2['language_code'])

language_df.head()
features = pd.concat([rating_df, language_df, df2['average_rating'], df2['ratings_count']], axis=1)

features.head()
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

features = min_max_scaler.fit_transform(features)
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')

model.fit(features)

dist, idlist = model.kneighbors(features)
def BookRecommender(book_name):

    book_list_name = []

    book_id = df2[df2['title'] == book_name].index

    book_id = book_id[0]

    for newid in idlist[book_id]:

        book_list_name.append(df2.loc[newid].title)

    return book_list_name
BookNames = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')

BookNames
BookNames = BookRecommender('The Lord of the Rings: Weapons and Warfare')

BookNames