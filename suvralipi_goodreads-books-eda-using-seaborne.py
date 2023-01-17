# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib as mpl

from matplotlib import pyplot as plt

import matplotlib.ticker as ticker



import seaborn as sns

import os

import warnings

df = pd.read_csv('../input/books.csv', error_bad_lines = False)
df.head()
df.describe()
null_count = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([null_count, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
most_read_lang = df['language_code'].value_counts().head(10)

ncount = df['bookID'].count()

plt.figure(figsize=(12,8))

sns.set(style="darkgrid")

ax = sns.barplot(most_read_lang.index, most_read_lang.values, alpha=0.9)

plt.title('Frequency Distribution of Books in various languages')

plt.ylabel('Number of Books', fontsize=12)

plt.xlabel('Language Code', fontsize=12)



# Make twin axis

ax2=ax.twinx()



# Switch so count axis is on right, frequency on left

ax2.yaxis.tick_left()

ax.yaxis.tick_right()



# Also switch the labels over

ax.yaxis.set_label_position('right')

ax2.yaxis.set_label_position('left')



ax2.set_ylabel('Frequency [%]')



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text



# Use a LinearLocator to ensure the correct number of ticks

ax.yaxis.set_major_locator(ticker.LinearLocator(11))



# Fix the frequency range to 0-100

ax2.set_ylim(0,100)

ax.set_ylim(0,ncount)



# And use a MultipleLocator to ensure a tick spacing of 10

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))



# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars

ax2.grid(None)
df_new = df[(df.language_code == 'eng') | (df.language_code == 'en-US') | (df.language_code == 'en-GB')]
h_labels = [x.replace('_', ' ').title() for x in 

            list(df_new.select_dtypes(include=['number', 'bool']).columns.values)]



fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(df_new.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
most_rated = df.sort_values('ratings_count',ascending=False).head(10).set_index('title')

plt.subplots(figsize=(12,8))

sns.set(style="darkgrid")



ax = most_rated['ratings_count'].sort_values().plot.barh(width=0.8,color=sns.color_palette('hls',12))

ax.set_xlabel("Total ratings count ", fontsize=15)

ax.set_ylabel("Book Title", fontsize=15)

ax.set_title("Top 10 most rated books",fontsize=20,color='black')



for i in ax.patches:

    ax.text(i.get_width()+5, i.get_y()+.3,str(round(i.get_width())), fontsize=15, color='black')

plt.show()
def categorize_rating(data):

    values = []

    for val in data.average_rating:

        if val>=0 and val<=1:

            values.append("Poor")

        elif val>1 and val<=2:

            values.append("Below Average")

        elif val>2 and val<=3:

            values.append("Average")

        elif val>3 and val<=4:

            values.append("Good")

        elif val>4 and val<=5:

            values.append("Excellent")

        else:

            values.append("NaN")

    print(len(values))

    return values
df_new['rating_category'] = categorize_rating(df_new)

df_new.head()
x = df_new['average_rating']

y = df_new['# num_pages']

z = np.log10(df_new['ratings_count']+1)

p = np.log10(df_new['text_reviews_count']+1)

c = df_new['rating_category']



sns.pairplot(pd.DataFrame(list(zip(x,y,z,p,c)),columns=['average_rating','No. of Pages', 'Rating Count', 'Text Review Count',

                                                      'rating_category']),

                           hue='rating_category',markers="o",palette="husl")
rating_value = df_new['rating_category'].value_counts()

print(rating_value)
plt.subplots(figsize=(10,8))

sns.set(style="whitegrid")

ax = sns.barplot(rating_value.index, rating_value.values, alpha=0.9, palette= sns.color_palette("pastel"))

plt.title('Frequency Distribution of various Ratings for English Books')

plt.ylabel('Number of Books', fontsize=12)

plt.xlabel('Rating Categories', fontsize=12)



# Make twin axis

ax2=ax.twinx()



# Switch so count axis is on right, frequency on left

ax2.yaxis.tick_left()

ax.yaxis.tick_right()



# Also switch the labels over

ax.yaxis.set_label_position('right')

ax2.yaxis.set_label_position('left')



ax2.set_ylabel('Frequency [%]')



for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

            ha='center', va='bottom') # set the alignment of the text



# Use a LinearLocator to ensure the correct number of ticks

ax.yaxis.set_major_locator(ticker.LinearLocator(11))



# Fix the frequency range to 0-100

ax2.set_ylim(0,100)

ax.set_ylim(0,ncount)



# And use a MultipleLocator to ensure a tick spacing of 10

ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))



# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars

ax2.grid(None)
df_new = df_new.drop(['isbn','isbn13','bookID'],axis=1)

df_new.head()
plt.subplots(figsize=(10,8))

sns.set(style="whitegrid")

ax = sns.stripplot(x="rating_category", y="ratings_count", data=df_new)
ax = sns.boxplot(x=df_new["rating_category"], y=np.log10(df_new["ratings_count"]+1))
plt.subplots(figsize=(10,6))

sns.violinplot(

    x='rating_category',

    y='# num_pages',

    data=df_new)
#ignoring data above 1000 pages

data_1 = df[df['# num_pages']<= 1000]
ax = sns.jointplot(x="average_rating", y="# num_pages", data = data_1, color = 'lightgreen')

ax.set_axis_labels("Average Rating", "Number of Pages")
most_popular_authors = df_new.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')



plt.subplots(figsize=(12,8))

sns.set(style="darkgrid")



ax = most_popular_authors['title'].sort_values().plot.barh(width=0.8,color=sns.color_palette('pastel',12))

ax.set_xlabel("Total no. of Books", fontsize=15)

ax.set_ylabel("Authors", fontsize=15)

ax.set_title("Top 10 Authors with higher book counts",fontsize=20,color='black')



for i in ax.patches:

    ax.text(i.get_width()+1, i.get_y()+.3,str(round(i.get_width())), fontsize=15, color='black')

plt.show()
## Authors with ratings more than 4.5

most_rated_authors = df_new[df_new["average_rating"]>4.5]

most_popular = most_rated_authors.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')



plt.subplots(figsize=(12,8))

sns.set(style="darkgrid")



ax = most_popular['title'].sort_values().plot.barh(width=0.8,color=sns.color_palette('colorblind',12))

ax.set_xlabel("Total no. of Books", fontsize=15)

ax.set_ylabel("Authors", fontsize=15)

ax.set_title("Top 10 Authors with higher Ratings",fontsize=20,color='black')



for i in ax.patches:

    ax.text(i.get_width(), i.get_y()+.3,str(round(i.get_width())), fontsize=15, color='black')

plt.show()