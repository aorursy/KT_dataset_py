import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
books = pd.read_csv('../input/books.csv', error_bad_lines=False)
books.head()
books.info()
### create the df

avg_rating_df = books.sort_values(['average_rating'], ascending=False).head(10)



### set figure size and axes

fig, ax = plt.subplots(figsize = (15,15))



### declare our x and y components

y_books = np.arange(len(avg_rating_df))

rating = avg_rating_df.average_rating.max()



### plot them and give necessary texts 

plt.style.use(['fivethirtyeight'])

ax.barh(y_books, rating, align='center', tick_label = avg_rating_df.title)

ax.invert_yaxis()  

ax.set_xlabel('Average Rating')

ax.set_title('Top 10 Books Based on Average Ratings')
### get df

most_rated = books.sort_values('ratings_count', ascending = False).head(10).set_index('title')



### set style

sns.set_style("darkgrid")



### initialize the matplotlib figure

fig, ax = plt.subplots(figsize=(10, 15))



### plot

sns.set_color_codes("pastel")

sns.barplot(most_rated['ratings_count'], most_rated.index, label="Total", color="b")

ax.set(ylabel="Books",xlabel="Rating Counts")

ax.set_title('Top 10 Books Based on User Ratings')
### get df

top_10_authors = books[books['average_rating']>=4]

top_10_authors = top_10_authors.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')



### set style

sns.set_style("ticks")



### set figure size

plt.figure(figsize=(15,10))



### plot

ax = sns.barplot(top_10_authors['title'], top_10_authors.index, palette='muted')

ax.set_xlabel("# Books")

ax.set_ylabel("Authors")

ax.set_title('Top 10 Most Rated Authors')

for i in ax.patches:

    ax.text(i.get_width()+.2, i.get_y()+0.3, str(round(i.get_width())), fontsize = 10, color = 'k')
### get df

lang_counts = pd.DataFrame(books.language_code.value_counts())

lang_counts = lang_counts.reset_index()

lang_counts = lang_counts.rename(columns={"index": "lang_code", "language_code": "counts"})

top_5_lang = lang_counts.sort_values(['counts'], ascending=False).head(5)



### set values

labels = top_5_lang.lang_code

counts = top_5_lang.counts

explode = (.05, 0, 0, 0,0) ### I want the most prominent language to be a little set apart when i view the pie.



### set figure and plot

fig, ax = plt.subplots(figsize = (20,20))

ax.pie(counts,explode=explode, labels=labels, autopct='%1.f%%', shadow=False, startangle=90)

ax.axis('equal')

ax.set_title('Top 5 Languages')



plt.show()