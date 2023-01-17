# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import missingno as msno

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)

df.head(1)
df.shape
df.info()
df.describe()
df.isnull().sum().sort_values()
cardinality={}

for col in df.columns:

    cardinality[col] = df[col].nunique()



cardinality
for row in df['publication_date']:

    if len(row) == 8:

        df['publication_date'].replace(row, ('0' + row), inplace=True)

        

for row in df['publication_date']:

    if row[1] == '/':

        df['publication_date'].replace(row, ('0' + row), inplace=True)



for row in df['publication_date']:

    if row[4] == '/':

        df['publication_date'].replace(row, (row[:3] + '0' + row[3:]), inplace=True)        
dates = pd.Series(df['publication_date'])

dates = pd.to_datetime(dates, errors='coerce')

df['publication_date'] = dates
df['publication_year'] = pd.to_datetime(df['publication_date']).dt.year

df['publication_year'] = df['publication_year'].fillna(0)

df['publication_year'] = df['publication_year'].astype(int)



df['publication_month'] = pd.to_datetime(df['publication_date']).dt.month

df['publication_month'] = df['publication_month'].fillna(0)

df['publication_month'] = df['publication_month'].astype(int)



df['pub_month_name'] = df['publication_date'].dt.strftime('%B')

df['pub_month_num'] = df['publication_date'].dt.strftime('%m')

df['publication_weekday'] = df['publication_date'].dt.strftime('%A')  

df['pub_date_num'] = df['publication_date'].dt.strftime('%d')

df['pub_year_month'] = df.publication_date.dt.to_period("M")
df = df[df['publication_year'] != 0]



df['publication_year'].isnull().value_counts()
bins = np.linspace(0, 5, 6)

group_names = ['btw 0 & 1', 'btw 1 & 2', 'btw 2 & 3', 'btw 3 & 4', 'btw 4 & 5']

df['ratings_binned'] = pd.cut(df['average_rating'], bins, labels=group_names, include_lowest=True)

#Remove Mary GrandPre from Harry Potter novels



df.replace('J.K. Rowling-Mary GrandPré', 'J.K. Rowling', inplace=True)
title_count = df.groupby(['title'])[['bookID']].agg('count').reset_index()

title_count.rename(columns={'bookID':'count'}, inplace=True)

title_count = title_count.sort_values('count', ascending=False)

title_count = title_count.reset_index().drop(['index'], axis=1)

title_count = title_count.head(20)
sns.set_context('poster')

plt.figure(figsize=(20,20))



x=title_count['count']

y=title_count['title']



ax = sns.barplot(x=x, y=y, palette='deep')





ax.set_title("MOST FREQUENTLY OCCURRING TITLES")

ax.set_xlabel("NUMBER OF OCCURANCES")

ax.set_ylabel("TITLES")

# plt.show()



most_rated_titles = df[['title', 'ratings_count', 'publication_year']].sort_values('ratings_count', ascending = False).reset_index()

most_rated_titles = most_rated_titles.drop(['index'], axis=1)

most_rated_titles = most_rated_titles.head(50)

most_rated_titles



# most_rated_titles = df[df['title'].isin(most_rated_titles['title'].unique())]

# most_rated_titles = most_rated_titles[most_rated_titles['title'].isin(most_rated_titles['title'].unique())]

# most_rated_titles
sns.set_context('poster')

plt.figure(figsize=(20,20))



x=most_rated_titles['ratings_count']

y=most_rated_titles['title']



sns.barplot(x=x, y=y, palette='deep')



plt.title("20 MOST RATED TITLES")

plt.xlabel("NUMBER OF RATINGS")

plt.ylabel("TITLES")

plt.show()
top_titles_list = most_rated_titles['title'].head(50).unique()



title_ratings = df.groupby(['title'])[['average_rating']].agg('mean').reset_index()

title_ratings =  title_ratings[title_ratings['title'].isin(top_titles_list)]

title_ratings =  title_ratings.sort_values('average_rating', ascending=False)

title_ratings =  title_ratings.reset_index().drop(['index'], axis=1)

title_ratings = title_ratings.head(25)

sns.set_context('poster')

plt.figure(figsize=(20,25))



x=title_ratings['average_rating']

y=title_ratings['title']



sns.barplot(x=x, y=y, palette='deep')



plt.title("HIGHEST RATED OF 25 MOST RATED TITLES")

plt.xlabel("AVERAGE RATING")

plt.ylabel("TITLES")

plt.show()
author_count = df[['authors']].value_counts().to_frame().reset_index()

author_count.rename(columns={0:'count'}, inplace=True)

author_count = author_count.head(20)



sns.set_context('poster')

plt.figure(figsize=(20,15))



x=author_count['count']

y=author_count['authors']



sns.barplot(x=x, y=y, palette='deep')



plt.title("20 MOST OCCURING AUTHORS")

plt.xlabel("NUMBER OF OCCURANCES")

plt.ylabel("AUTHORS")

plt.show()
most_rated_auth = author_count['authors'].head(20).unique()



auth_avg_rating = df.groupby(['authors'])[['average_rating']].agg('mean').reset_index()

auth_avg_rating = auth_avg_rating[auth_avg_rating['authors'].isin(most_rated_auth)]

auth_avg_rating = auth_avg_rating.sort_values('average_rating', ascending=False)

auth_avg_rating = auth_avg_rating.reset_index().drop(['index'], axis=1)

auth_avg_rating = auth_avg_rating.head(20)

sns.set_context('poster')

plt.figure(figsize=(20,15))



x=auth_avg_rating['average_rating']

y=auth_avg_rating['authors']



sns.barplot(x=x, y=y, palette='deep')



plt.title("HIGHEST RATED OF 20 MOST OCCURING AUTHORS")

plt.xlabel("AVERAGE RATING")

plt.ylabel("AUTHORS")

plt.show()
book_years = df.groupby(['publication_year'])[['bookID']].agg('count')

book_years.rename(columns={'bookID' : 'count'}, inplace=True)

book_years = book_years.sort_values('count', ascending=False).reset_index()

book_years = book_years.head(20)

book_years = book_years.sort_values('publication_year').reset_index().drop(['index'], axis=1)

sns.set_context('poster')

plt.figure(figsize=(20,15))



x=book_years['publication_year']

y=book_years['count']



chart = sns.barplot(x=x, y=y, palette='deep')



plt.title("YEARS WITH THE MOST LISTED BOOKS")

plt.xlabel("YEARS")

plt.xticks(rotation=45,  horizontalalignment='center',fontweight='light',fontsize='small')

plt.ylabel("NUMBER OF BOOKS")

plt.show()



rating_month = df.groupby(['publication_month', 'pub_month_name'])[['bookID']].agg('count')

rating_month = rating_month.sort_values('bookID', ascending=False).reset_index()

rating_month.rename(columns={'bookID':'book_count'}, inplace=True)

rating_month = rating_month.head(20).sort_values('publication_month')
sns.set_context('poster')

plt.figure(figsize=(20,15))



x=rating_month['pub_month_name']

y=rating_month['book_count']



chart = sns.barplot(x=x, y=y, palette='deep')



plt.title("MONTHS WITH THE MOST LISTED BOOKS")

plt.xlabel("MONTHS")

plt.xticks(rotation=45,  horizontalalignment='center',fontweight='light',fontsize='small')

plt.ylabel("NUMBER OF BOOKS")

plt.show()

pub_ratings_count = df.groupby(['publisher'])[['ratings_count']].agg('count')

pub_ratings_count = pub_ratings_count.sort_values('ratings_count', ascending=False).reset_index()

pub_ratings_count = pub_ratings_count.head(20)
sns.set_context('poster')

plt.figure(figsize=(20,15))



x=pub_ratings_count['ratings_count']

y=pub_ratings_count['publisher']



sns.barplot(x=x, y=y, palette='deep')



plt.title("20 MOST RATED PUBLISHERS")

plt.xlabel("NUMBER OF RATINGS")

plt.ylabel("PUBLISHERS")

plt.show()
most_rated_pub = pub_ratings_count['publisher'].head(20).unique()



pub_avg_ratings = df.groupby(['publisher'])[['average_rating']].agg('mean').reset_index()

pub_avg_ratings = pub_avg_ratings[pub_avg_ratings['publisher'].isin(most_rated_pub)]

pub_avg_ratings = pub_avg_ratings.sort_values('average_rating', ascending=False)

pub_avg_ratings = pub_avg_ratings.reset_index().drop(['index'], axis=1)
sns.set_context('poster')

plt.figure(figsize=(15,20))



x=pub_avg_ratings['average_rating']

y=pub_avg_ratings['publisher']



sns.barplot(x=x, y=y, palette='deep')



plt.title("Average Ratings of the 20 Most Rated Publishers")

plt.xlabel("Average Rating")

plt.ylabel("Publishers")

plt.show()
languages = df.groupby(['language_code'])[['title']].agg('count')

languages = languages.rename(columns={'title':'count'}).reset_index()

languages = languages.sort_values('count', ascending=False)



sns.set_context('talk')

plt.figure(figsize=(15,10))



sns.barplot(x="language_code", y='count', data=languages, palette='deep')



plt.title("DISTRIBUTION OF LANGUAGES")

plt.xlabel("LANGUAGES")

plt.xticks(rotation=45,  horizontalalignment='center',fontweight='light',fontsize='12')



plt.ylabel("COUNT")

plt.show()
fig = px.pie(df, values='average_rating', names='ratings_binned')

fig.show()
fig = px.scatter(x=df['average_rating'], y=df['  num_pages'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: AVERAGE RATING & NUM OF PAGES',

                  yaxis_title='NUM OF PAGES',

                  xaxis_title='AVERAGE RATING', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()
fig = px.scatter(x=df['average_rating'], y=df['ratings_count'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: AVERAGE RATING & RATINGS COUNT',

                  yaxis_title='ratings_count',

                  xaxis_title='AVERAGE RATING', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()
sample = df[df['ratings_count']<20000]



fig = px.scatter(x=sample['average_rating'], y=sample['ratings_count'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: AVERAGE RATING & RATINGS COUNT',

                  yaxis_title='ratings_count',

                  xaxis_title='AVERAGE RATING', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()
ratings_reviews = df[['text_reviews_count', 'ratings_binned']].sort_values('text_reviews_count', ascending=False)

# ratings_reviews
fig = px.scatter(x=ratings_reviews['ratings_binned'], y=ratings_reviews['text_reviews_count'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: AVERAGE RATING & TEXT REVIEWS COUNT',

                  yaxis_title='TEXT REVIEWS COUNT',

                  xaxis_title='AVERAGE RATING BINS', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()
fig = px.scatter(x=df['average_rating'], y=df['text_reviews_count'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: AVERAGE RATING & TEXT REVIEWS COUNT',

                  yaxis_title='TEXT REVIEWS COUNT',

                  xaxis_title='AVERAGE RATING', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()
fig = px.scatter(x=df['average_rating'], y=df['publication_year'])



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='CORRELATION: RATING & YEAR OF PUBLICATION',

                  yaxis_title='YEAR OF PUBLICATION',

                  xaxis_title='AVERAGE RATING', 

                  title_x=0.5,

                  width = 750,

                  height=500)

fig.show()