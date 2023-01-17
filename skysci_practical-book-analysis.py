import numpy as np # number processing 

import pandas as pd # data processing 

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization

import os # directory access



# return list of files in directory 'input'

print(os.listdir('../input')) 

# load dataset

df = pd.read_csv('../input/books.csv', error_bad_lines=False) 
# return number of rows and columns

df.shape 
# check for missing values

df.count()
# check each column's data type

df.dtypes
# summary of statistics

df.describe()
# return first 5 rows

df.head() 
# rename columns

df.rename(columns={'average_rating':'avg_rating',

                   '# num_pages':'num_pages',

                   'language_code':'lang_code'},inplace=True) 

df.columns
# find out what and how many language codes are there

print(df['lang_code'].unique())

print('\n Total language codes:', len(df['lang_code'].unique()))
# top 10 languages for books

langs = df['lang_code'].value_counts().head(10)

plt.figure(figsize=(15,6))

sns.barplot(x=langs, y=langs.index) # horizontal bar plot

sns.despine() # remove line to the top and right of chart

sns.despine(left=True, bottom=True) # remove line to the bottom and left of chart

plt.title('Top 10 Languages By Number of Books Written', fontsize=20, fontweight='bold')

plt.xlabel('Number of Books', fontsize=12, fontstyle='italic') 
# books written in all variants of English

eng_books = df[(df['lang_code'] == 'eng') | (df['lang_code'] == 'en-US') | (df['lang_code'] == 'en-GB') 

               | (df['lang_code'] == 'en-CA')]



# plot a pie chart to show the percentage of English books out of all total books

sizes = [eng_books.shape[0], df.shape[0]] 

labels = ['English', 'Other Languages']

colors = ['lightblue', 'lightcoral']

explode=(0.1, 0) # explode the first slice of the pie

plt.pie(sizes, labels=labels, colors=colors, explode=explode, textprops=dict(fontsize=16), autopct='%1.0f%%', shadow=True, startangle=90)

plt.title('English vs Other Languages', fontsize=20, fontweight='bold')

plt.axis('equal')
# plot books against average rating

sns.distplot(a=df['avg_rating'], kde=False)

sns.despine()

sns.despine(left=True, bottom=True)
# correlation between average rating, ratings count, and text reviews count

sns.set_style('whitegrid')

sns.scatterplot(x=df['avg_rating'], y=df['ratings_count'], hue=df['text_reviews_count'])

sns.despine()

sns.despine(left=True, bottom=True)
# find outliers

df[df['ratings_count'] > 4000000]
# find Twilight sequels

df[df['authors'] == 'Stephenie Meyer'].sort_values(['ratings_count', 'text_reviews_count'], ascending=False)
# find Harry Potter sequels

df[(df['authors'] == 'J.K. Rowling-Mary GrandPré') | 

   (df['authors'] == 'J.K. Rowling')].sort_values(['ratings_count', 'text_reviews_count'], ascending=False)