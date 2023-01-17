# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')
df.head()
df = df[['airline_sentiment', 'airline', 'text']]
pd.set_option('display.max_colwidth',200)
df.head()
df.isnull().sum()
import re
def remove_links(tweet):

    tweet = re.sub(r'http\S+', '', tweet) # remove http links

    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links

    return tweet
df['text'] = df['text'].apply(lambda x: remove_links(x))
df.head(5)
def remove_users(tweet):

    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet

    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at

    return tweet
df['text'] = df['text'].apply(lambda x: remove_users(x))
df.head(5)
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
df['scores'] = df['text'].apply(lambda x: sid.polarity_scores(x))
df.head()
df['compound'] = df['scores'].apply(lambda x: x['compound'])
df.head()
def comp_sc(x):

    if x>=0.05:

        return 'positive'

    elif x<= -0.05:

        return 'negative'

    else:

        return 'neutral'
df['comp_score'] = df['compound'].apply(lambda x: comp_sc(x))
df.head(5)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(df['airline_sentiment'], df['comp_score'])
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(20,12))

sns.countplot(x= 'airline', hue = 'comp_score', data = df)