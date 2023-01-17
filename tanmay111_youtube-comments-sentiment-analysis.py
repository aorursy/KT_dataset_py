import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
pd.set_option('display.max_columns',None)
US_comments = pd.read_csv('../input/youtube/UScomments.csv', error_bad_lines=False)
US_videos = pd.read_csv('../input/youtube/USvideos.csv', error_bad_lines=False)
US_videos.head()
US_videos.shape
US_videos.nunique()
US_videos.info()
US_videos.head()
US_comments.head()
US_comments.shape
US_comments.isnull().sum()
US_comments.dropna(inplace=True)
US_comments.isnull().sum()
US_comments.shape
US_comments.nunique()
US_comments.info()
US_comments.drop(41587, inplace=True)
US_comments = US_comments.reset_index().drop('index',axis=1)
US_comments.likes = US_comments.likes.astype(int)

US_comments.replies = US_comments.replies.astype(int)
US_comments.head()
US_comments['comment_text'] = US_comments['comment_text'].str.replace("[^a-zA-Z#]", " ")
US_comments['comment_text'] = US_comments['comment_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
US_comments['comment_text'] = US_comments['comment_text'].apply(lambda x:x.lower())
tokenized_tweet = US_comments['comment_text'].apply(lambda x: x.split())

tokenized_tweet.head()
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
wnl = WordNetLemmatizer()
tokenized_tweet.apply(lambda x: [wnl.lemmatize(i) for i in x if i not in set(stopwords.words('english'))]) 

tokenized_tweet.head()
for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
US_comments['comment_text'] = tokenized_tweet
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
US_comments['Sentiment Scores'] = US_comments['comment_text'].apply(lambda x:sia.polarity_scores(x)['compound'])
US_comments.head()
US_comments['Sentiment'] = US_comments['Sentiment Scores'].apply(lambda s : 'Positive' if s > 0 else ('Neutral' if s == 0 else 'Negative'))
US_comments.head()
US_comments.Sentiment.value_counts()
videos = []

for i in range(0,US_comments.video_id.nunique()):

    a = US_comments[(US_comments.video_id == US_comments.video_id.unique()[i]) & (US_comments.Sentiment == 'Positive')].count()[0]

    b = US_comments[US_comments.video_id == US_comments.video_id.unique()[i]]['Sentiment'].value_counts().sum()

    Percentage = (a/b)*100

    videos.append(round(Percentage,2))
Positivity = pd.DataFrame(videos,US_comments.video_id.unique()).reset_index()
Positivity.columns = ['video_id','Positive Percentage']
Positivity.head()
channels = []

for i in range(0,Positivity.video_id.nunique()):

    channels.append(US_videos[US_videos.video_id == Positivity.video_id.unique()[i]]['channel_title'].unique()[0])
Positivity['Channel'] = channels
Positivity.head()
Positivity[Positivity['Positive Percentage'] == Positivity['Positive Percentage'].max()]
Positivity[Positivity['Positive Percentage'] == Positivity['Positive Percentage'].min()]
all_words = ' '.join([text for text in US_comments['comment_text']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
all_words_posi = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Positive']])
wordcloud_posi = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_posi)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud_posi, interpolation="bilinear")

plt.axis('off')

plt.show()
all_words_nega = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Negative']])
wordcloud_nega = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_nega)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud_nega, interpolation="bilinear")

plt.axis('off')

plt.show()
all_words_neu = ' '.join([text for text in US_comments['comment_text'][US_comments.Sentiment == 'Neutral']])
wordcloud_neu = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_neu)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud_neu, interpolation="bilinear")

plt.axis('off')

plt.show()