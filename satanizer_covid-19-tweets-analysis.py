import io

import random

import string

import warnings

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import warnings

warnings.filterwarnings('ignore')





import nltk

from nltk.tokenize import sent_tokenize

from nltk.corpus import words

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *





# sklearn imports

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics





# python imports

import re

import json

import os

from collections import Counter

import datetime as dt





# Visualization

from matplotlib import pyplot as plt

from matplotlib import ticker

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from wordcloud import WordCloud

from tqdm import tqdm_notebook





# Saving models

import pickle
data_dir = '../input/covid19tweets/data'

tweets = []

for file in sorted(os.listdir(data_dir)):

    tweets.append(pd.read_csv(data_dir + '/' + file, lineterminator = '\n'))

#     print(file)



df = pd.concat(tweets)

df.tail()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(111)

ax.set(title='Temporal tweet frequency worldwide', xlabel='Time', ylabel='Tweet frequency per hour')

plt.hist(pd.to_datetime(df.created_at), bins = 24*9, color = 'b')

plt.show()
text_en = df['text']
text_en_lr = text_en.apply(lambda x: re.sub(r"https\S+", "", str(x)))

text_en_lr.head()
text_en_lr_lc = text_en_lr.apply(lambda x: x.lower())

text_en_lr_lc.head()
text_en_lr_lc_pr = text_en_lr_lc.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

text_en_lr_lc_pr.head()
stop_words = set(stopwords.words('english'))

stop_words.update(['#coronavirus', '#coronavirusoutbreak', '#coronavirusPandemic', '#covid19', '#covid_19', '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19'])



text_en_lr_lc_pr_sr = text_en_lr_lc_pr.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

text_en_lr_lc_pr_sr.head()
word_list = [word for line in text_en_lr_lc_pr_sr for word in line.split()]

word_list[:5]
sns.set(style="darkgrid")

counts = Counter(word_list).most_common(50)

counts_df = pd.DataFrame(counts)

counts_df

counts_df.columns = ['word', 'frequency']



fig, ax = plt.subplots(figsize = (12, 12))

ax = sns.barplot(y="word", x='frequency', ax = ax, data=counts_df)

plt.savefig('wordcount_bar.png')
wordcloud = WordCloud(

    background_color='black',

    max_words=50,

    max_font_size=40, 

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(word_list))





plt.figure(figsize = (12, 10), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad = 0)



plt.savefig('wordcloud.png')



sid = SentimentIntensityAnalyzer()

sentiment_scores = text_en_lr_lc_pr_sr.apply(lambda x: sid.polarity_scores(x))

sent_scores_df = pd.DataFrame(list(sentiment_scores))

sent_scores_df.tail()
sent_scores_df['val'] = sent_scores_df['compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))

sent_scores_df.head()
sent_counts = pd.DataFrame.from_dict(Counter(sent_scores_df['val']), orient = 'index').reset_index()

sent_counts.columns = ['sentiment', 'count']



sns.barplot(y="count", x='sentiment', data=sent_counts)

plt.savefig('sentiment.png')
sentiments_time_df = pd.DataFrame()

sentiments_time_df['time'] = df['created_at']

sentiments_time_df['polarity'] = sent_scores_df['compound']

sentiments_time_df.index = pd.to_datetime(sentiments_time_df['time'])





ot = sentiments_time_df.sample(frac=.001)

ot['time'] = pd.to_datetime(ot['time'])

ot.index = pd.to_datetime(ot['time'])

ot.sort_index(inplace=True)

ot['expanding'] = ot['polarity'].expanding().mean()

ot['rolling'] = ot['polarity'].rolling('1h').mean()



fig = plt.figure(figsize=(20,5))

ax = fig.add_subplot(111)

ax.scatter(ot['time'],ot['polarity'], label='Tweet Sentiment', s = 10, color = 'y')

ax.plot(ot['time'],ot['rolling'], color ='r', label='Rolling Mean', linewidth = 5)

ax.plot(ot['time'],ot['expanding'], color='b', label='Expanding Mean', linewidth = 5)

ax.set_xlim([dt.date(2020,5,1),dt.date(2020,5,9)])

ax.set(title='Tweet Sentiments over Time', xlabel='Date', ylabel='Sentiment polarity')

ax.legend(loc='best')

fig.tight_layout()

plt.savefig('temporal_sentiments.png')

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(111)

ax.set(title='Tweet Sentiments distribution', xlabel='polarity', ylabel='frequency')

sns.distplot(sentiments_time_df['polarity'], bins=30, ax=ax)

# plt.show()

plt.savefig('sentiment_distribution.png')
polar_tweets_df = pd.DataFrame()

polar_tweets_df['tweet'] = text_en_lr_lc_pr_sr

polar_tweets_df['polarity'] = sent_scores_df['val']



positive = polar_tweets_df[polar_tweets_df['polarity'] == 'positive']['tweet']

negative = polar_tweets_df[polar_tweets_df['polarity'] == 'negative']['tweet']

neutral = polar_tweets_df[polar_tweets_df['polarity'] == 'neutral']['tweet']



positive_list = [word for line in positive for word in line.split()]

negative_list = [word for line in negative for word in line.split()]

neutral_list = [word for line in neutral for word in line.split()]



positive_cloud = WordCloud(

    background_color='black',

    max_words=50,

    max_font_size=40, 

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(positive_list))



negative_cloud = WordCloud(

    background_color='black',

    max_words=50,

    max_font_size=40, 

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(negative_list))



neutral_cloud = WordCloud(

    background_color='black',

    max_words=50,

    max_font_size=40, 

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(neutral_list))





fig, axs = plt.subplots(2, 2, figsize = (20, 12))

# fig.suptitle('Clouds of polar words', fontsize = 30)

fig.tight_layout(pad = 0)



axs[0, 0].imshow(positive_cloud)

axs[0, 0].set_title('Words from positive tweets', fontsize = 20)

axs[0, 0].axis('off')

# axs[0, 0].tight_layout(pad = 1)



axs[0, 1].imshow(negative_cloud)

axs[0, 1].set_title('Words from negative tweets', fontsize = 20)

axs[0, 1].axis('off')

# axs[0, 1].tight_layout(pad = 1)



axs[1, 0].imshow(neutral_cloud)

axs[1, 0].set_title('Words from neutral tweets', fontsize = 20)

axs[1, 0].axis('off')

# axs[1, 0].tight_layout(pad = 1)



axs[1, 1].imshow(wordcloud)

axs[1, 1].set_title('Words from all tweets', fontsize = 20)

axs[1, 1].axis('off')

# axs[1, 0].tight_layout(pad = 1)

plt.savefig('joint_cloud.png')


