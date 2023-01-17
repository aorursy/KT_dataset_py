import os

import gc

import sys



import numpy as np

import pandas as pd



import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer



import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud
train_df = pd.read_csv('../input/train.csv')
train_df.head()
revenues = train_df['revenue']
budgets = train_df['budget']
sns.jointplot(x=budgets, y=revenues, dropna=True, color='blueviolet', kind='reg')

plt.show()
plot = sns.jointplot(x='popularity', y='revenue', data=train_df, dropna=True, color='orangered', kind='reg') 
fig, ax = plt.subplots(figsize=(15, 15))

ax.tick_params(axis='both', labelsize=12)

plt.title('Original Language and Revenue', fontsize=20)

plt.xlabel('Revenue', fontsize=16)

plt.ylabel('Original Language', fontsize=16)

sns.boxplot(ax=ax, x='revenue', y='original_language', data=train_df, showfliers=False, orient='h')

plt.show()
plt.figure(figsize = (12, 8))

text = ' '.join(train_df['original_language'])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Languages', fontsize=20)

plt.axis("off")

plt.show()
genres = []

repeated_revenues = []

for i in range(len(train_df)):

  if train_df['genres'][i] == train_df['genres'][i]:

      movie_genre = [genre['name'] for genre in eval(train_df['genres'][i])]

      genres.extend(movie_genre)

      repeated_revenues.extend([train_df['revenue'][i]]*len(movie_genre))

  

genre_df = pd.DataFrame(np.zeros((len(genres), 2)))

genre_df.columns = ['genre', 'revenue']

genre_df['genre'] = genres

genre_df['revenue'] = repeated_revenues
fig, ax = plt.subplots(figsize=(15, 15))

ax.tick_params(axis='both', labelsize=12)

plt.title('Genres and Revenue', fontsize=20)

plt.xlabel('revenue', fontsize=16)

plt.ylabel('genre', fontsize=16)

sns.boxplot(ax=ax, x=repeated_revenues, y=genres, showfliers=False, orient='h')

plt.show()
plt.figure(figsize = (12, 8))

text = ' '.join(genres)

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=2000, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top Genres', fontsize=30)

plt.axis("off")

plt.show()
def sentiment(x):

  if type(x) == str:

    return SIA.polarity_scores(x)

  else:

    return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
SIA = SentimentIntensityAnalyzer()

overview_sentiments = train_df['overview'].apply(lambda x: sentiment(x))

tagline_sentiments = train_df['tagline'].apply(lambda x: sentiment(x))
neutralities = [sentiment['neu'] for sentiment in tagline_sentiments]

negativities = [sentiment['neg'] for sentiment in overview_sentiments]

compound = [sentiment['compound'] for sentiment in overview_sentiments]
sns.jointplot(x=negativities, y=revenues, dropna=True, color='mediumvioletred', kind='scatter')

plt.show()
sns.jointplot(x=neutralities, y=revenues, dropna=True, color='mediumblue', kind='reg')

plt.show()
sns.jointplot(x=compound, y=revenues, dropna=True, color='maroon', kind='reg')

plt.show()
lengths = train_df['tagline'].apply(lambda x: len(str(x)))
sns.jointplot(x=lengths, y=revenues, dropna=True, color='crimson')