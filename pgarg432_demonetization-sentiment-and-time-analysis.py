# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
import re
import sys
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tweets = pd.read_csv('../input/demonetization-tweets.csv', encoding = 'ISO-8859-1')
tweets.head()
# Preprocessing del RT @blablabla:
pd.options.mode.chained_assignment = None 

# add tweetos first part
for i in range(len(tweets['text'])):
    try:
        tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()[2:]])
    except AttributeError:    
        tweets['text'][i] = 'other'
tweets.head()
def wordcloud_by_province(tweets):
    stopwords = set(STOPWORDS)
    stopwords.add("https")
    #below word are contained many times in the tweets text and are of no significance for getting insights from the analysis.
    stopwords.add("00A0")
    stopwords.add("00BD")
    stopwords.add("00B8")
    stopwords.add("ed")
    stopwords.add("demonetization")
    stopwords.add("Demonetization co")
    #Narendra Modi is the Prime minister of India
    stopwords.add("lakh")
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets['text'].str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Demonetization")

wordcloud_by_province(tweets)
#1.) 1. What percentage of tweets is negative, positive or neutral?
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from nltk import tokenize
sid = SentimentIntensityAnalyzer()

tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets.head()
tweets.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")
#2.) What are the most famous/re-tweeted tweets? 
tweets.sort_values('retweetCount', ascending = False).head(10)['text']
#3.) Create stacked chart (Retweets, Total Tweets) showing “„Hour of the Day Trends” TweetCount Vs Hour. 
tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour
tweets_hour = tweets.groupby(['hour'])['retweetCount'].sum()
import seaborn as sns
tweets_hour.transpose().plot(kind='line',figsize=(6.5, 4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('The number of retweet by hour', bbox={'facecolor':'0.8', 'pad':0})
tweets['tcount'] = 1 #new col to find totoal tweets per hour
tweets.groupby(["hour"]).sum().reset_index()['tcount'].transpose().plot(kind='line',figsize=(6.5, 4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('The number of tweet by hour', bbox={'facecolor':'0.8', 'pad':0})
#5.) Create Bar chart showing Tweet counts Device wise (twitter for Android, twitter Web client, Twitter for iPhone, Facebook, Twitter for iPad, etc.)
tweets['statusSource_new'] = ''

for i in range(len(tweets['statusSource'])):
    m = re.search('(?<=>)(.*)', tweets['statusSource'][i])
    try:
        tweets['statusSource_new'][i]=m.group(0)
    except AttributeError:
        tweets['statusSource_new'][i]=tweets['statusSource'][i]
        
#print(tweets['statusSource_new'].head())   

tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)
tweets['statusSource_new2'] = ''

for i in range(len(tweets['statusSource_new'])):
    if tweets['statusSource_new'][i] not in ['Twitter for Android ','Twitter Web Client ','Twitter for iPhone ', 'Mobile Web (M5) ', 'Facebook']:
        tweets['statusSource_new2'][i] = 'Others'
    else:
        tweets['statusSource_new2'][i] = tweets['statusSource_new'][i] 
#print(tweets['statusSource_new2'])       

tweets_by_type2 = tweets.groupby(['statusSource_new2'])['retweetCount'].sum()
tweets_by_type2.rename("",inplace=True)
explode = (0, 0, 0, 0, 1.0)
tweets_by_type2.transpose().plot(kind='bar')
plt.title("Number of tweetcount by Source")
plt.xlabel('Source')
plt.ylabel('Tweet Count')
tweets_by_type2.rename("",inplace=True)
explode = (0, 0, 0, 0, 1.0)
tweets_by_type2.transpose().plot(kind='pie',figsize=(6.5, 4),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=6, borderaxespad=0.)
plt.title('Number of tweetcount by Source', bbox={'facecolor':'0.8', 'pad':5})
#6.) Most Popular 10 Users
popular_users = tweets.groupby('screenName').sum()['retweetCount'].sort_values(ascending = False)
popular_users.head()
