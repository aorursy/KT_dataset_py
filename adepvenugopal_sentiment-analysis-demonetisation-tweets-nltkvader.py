import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

from nltk import tokenize

#Visualization packages

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_colwidth', 500)
tweets=pd.read_csv("../input/demonetization-tweets.csv",encoding = "ISO-8859-1")
tweets.head()
tweets.shape
from wordcloud import WordCloud



def wc(data,bgcolor,title):

    plt.figure(figsize = (50,50))

    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)

    wc.generate(' '.join(data))

    plt.imshow(wc)

    plt.axis('off')
wc(tweets['text'],'black','Common Words' )
sid = SentimentIntensityAnalyzer()

sid.polarity_scores(tweets.text[1])
tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])#extract compound score

tweets['sentiment_neutral']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neu'])#extract neutral score

tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])#extract negative score

tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])#extract positive score

tweets['sentiment_type']='' #initialize sentiment_type column

tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'

tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'

tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

tweets.head()
tweets.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")
tweets.sentiment_compound_polarity.hist(bins=50)
tweets[['screenName','text','sentiment_compound_polarity']][tweets.sentiment_compound_polarity > 0]
wc(tweets.text[tweets.sentiment_compound_polarity > 0.8],'black','Common Words' )
tweets[['screenName','text','sentiment_compound_polarity']][tweets.sentiment_compound_polarity < 0]
wc(tweets.text[tweets.sentiment_compound_polarity < -0.8],'black','Common Words' )
tweets[['screenName','text','sentiment_compound_polarity']][tweets.sentiment_compound_polarity == 0]
wc(tweets.text[tweets.sentiment_compound_polarity == 0],'black','Common Words' )
tweets.screenName.value_counts(sort=True, ascending=False).head(10)
tweets[['screenName','sentiment_compound_polarity']].groupby('screenName').sum().sort_values(by=['sentiment_compound_polarity'])
tweets[['screenName','sentiment_compound_polarity']].groupby('screenName').sum().sort_values(by=['sentiment_compound_polarity'], ascending=False).head(10)
tweets[['screenName','text','created','sentiment_compound_polarity']][tweets.screenName == 'guna5555']
tweets[['screenName','sentiment_compound_polarity']].groupby('screenName').sum().sort_values(by=['sentiment_compound_polarity'], ascending=True).head(10)
tweets[['screenName','text','created','sentiment_compound_polarity']][tweets.screenName == 'Stupidosaur']