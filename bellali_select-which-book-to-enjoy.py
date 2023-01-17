import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load data file

df = pd.read_csv('../input/books.csv', error_bad_lines=False)

df.sample(5)
df.shape
# check out null data

df.info()
# check out duplicated data

df.duplicated().sum()
# Check out the unique data of some columns

df['authors'].unique()
# Returns valid descriptive statistics for each column of data

df.describe()
# copy the data

data = df.copy()
# change "# num_pages" to "num_pages"

data.rename(columns = {'# num_pages': 'num_pages'}, inplace=True)
# drop the data with less than 300 ratings counts

data = data[data['ratings_count'] > 299]
# find the book data with 0 page

data[data['num_pages'] == 0]
# fill the "num_pages" with right number

data.loc[data['bookID'] == 2874, 'num_pages'] = 209

data.loc[data['bookID'] == 17983, 'num_pages'] = 306

data.loc[data['bookID'] == 32974, 'num_pages'] = 1020 # the sum pages of Nancy Drew: #1-6
# replace the 0 page by the average page of all books

data.loc[data['num_pages'] == 0, 'num_pages'] = data['num_pages'].mean()
data['num_pages'] = data['num_pages'].astype(int)
# dorp the useless columns

data.drop(['bookID', 'isbn', 'isbn13'], axis=1, inplace=True)
# select the 1st author

data['authors'] = data['authors'].apply(lambda x: x.split("-")[0])
# select top 10 rating books

top10_rating = data.sort_values('average_rating', ascending=False)[:10]
top10_rating
sns.set(style="white", context="talk")

fig = plt.figure(figsize=(13,12))

x = top10_rating['average_rating']

y = top10_rating['title']

sns.barplot(x=x, y=y, palette="GnBu_d")

plt.title('The top 10 rating books', fontsize=24)

plt.show();
# select the top 10 books with highest text_reviews_count 

top10_pop = data.sort_values('text_reviews_count', ascending=False)[:10]

top10_pop
sns.set(style="white", context="talk")

fig = plt.figure(figsize=(13,12))

x = top10_pop['text_reviews_count']

y = top10_pop['title']

sns.barplot(x=x, y=y, palette="GnBu_d")

plt.title('The top 10 popular books', fontsize=24)

plt.show();
# average_rating & text_reviews_count

sns.set(style="darkgrid")

sns.jointplot(y = "text_reviews_count", x = "average_rating", data=data, kind="reg", space=0.5)

plt.title('The Scatter of Average Rating and Text Reviews Count',fontsize=15)

plt.show();
# average_rating & ratings_count

sns.set(style="darkgrid")

sns.jointplot(y = "ratings_count", x = "average_rating", data=data, kind="reg", space=0.5)

plt.title('The Scatter of Average Rating and Ratings Count',fontsize=15)

plt.show();
# average_rating & num_pages

sns.set(style="darkgrid")

sns.jointplot(y = "num_pages", x = "average_rating", data=data, kind="reg", space=0.5)

plt.title('The Scatter of Average Rating and Num Pages',fontsize=15)

plt.show();
fig = plt.figure(figsize=(9,8))

sns.distplot(data['average_rating'])

plt.title('The Distribution across Avrage Rating', fontsize=15)

plt.vlines(data['average_rating'].mean(),ymin = 0,ymax = 1.75,color = 'black')

plt.vlines(data['average_rating'].median(),ymin = 0,ymax = 1.75,color = 'red')

plt.vlines(data['average_rating'].mode(),ymin = 0,ymax = 1.75,color = 'yellow')

plt.legend()

plt.show()
# select data

top10_production = data['authors'].value_counts()[:10]

top10_production = pd.DataFrame(top10_production)

top10_production
sns.set(style="white", context="talk")

fig = plt.figure(figsize=(13,12))

x = top10_production['authors']

y = top10_production.index

sns.barplot(x=x, y=y, palette="GnBu_d")

plt.title('The top 10 highest production authors', fontsize=24)

plt.show();
# select data

top10_rating_authors = pd.DataFrame(data.groupby('authors')['average_rating'].mean().sort_values(ascending=False)[:10])

top10_rating_authors
sns.set(style="white", context="talk")

fig = plt.figure(figsize=(13,12))

x = top10_rating_authors['average_rating']

y = top10_rating_authors.index

sns.barplot(x=x, y=y, palette="GnBu_d")

plt.title('The top 10 authors whose books have the highest rating', fontsize=24)

plt.show();
production = pd.DataFrame(data['authors'].value_counts())

production.reset_index(inplace=True)

production.rename(columns={'index': 'authors', 'authors': 'production_counts'}, inplace=True)

production.head()
rating = pd.DataFrame(data.groupby('authors')['average_rating'].mean())

rating.reset_index(inplace=True)
data_1 = rating.merge(production, how='inner')

data_1.head()
sns.set(style="darkgrid")

sns.jointplot(y = "production_counts", x = 'average_rating', data=data_1, kind="reg", space=0.5)

plt.title('The Scatter of Average Rating(authors) and Productions',fontsize=15)

plt.show();
# select data

data_2 = data[data['authors'].isin(['George R.R. Martin', 'J.R.R. Tolkien', 'J.K. Rowling'])]

data_2 = data_2.query('language_code == "eng"')

data_2.head()
data_2.describe()
# classify the rating level

bin_edges = [3.5, 4.0, 4.5, 5.0]

bin_names = ['low', 'medium', 'high']

data_2['rating_levels'] = pd.cut(data_2['average_rating'], bin_edges, labels=bin_names)
data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))

data_3
# Pie chart

labels = 'high', 'low', 'medium'

explode = (0.1, 0, 0)

fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (12,4))



# George R.R. Martin

ax1.pie(data_3.iloc[0], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.set_title('George R.R. Martin')

ax1.axis('equal')

# J.K. Rowling

ax2.pie(data_3.iloc[1], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax2.set_title('J.K. Rowling')

ax2.axis('equal')

# J.R.R. Tolkien

ax3.pie(data_3.iloc[2], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax3.set_title('J.R.R. Tolkien')

ax3.axis('equal')



plt.show()
# compare "Harry Potter", "The Lord of the Rings" and "A Song of Ice and Fire"

data_4 = data_2[data_2['title'].str.contains('Harry Potter|The Lord of the Rings|A Song of Ice and Fire')]
data_4.groupby('authors')['average_rating'].mean().sort_values(ascending=False)