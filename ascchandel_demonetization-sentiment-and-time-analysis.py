# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
 
    # Importing neccessary libraries
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
demo_tweet=pd.read_csv('../input/demonetization-tweets.csv',encoding='ISO-8859-1')
demo_tweet.head()
#1. Remove HTML tags using BeautifulSoup
demo_tweet['text'] = [BeautifulSoup(text.lower(),"lxml").get_text() for text in demo_tweet['text']]
demo_tweet['text'].head()
from nltk.sentiment import vader
from nltk.sentiment.util import *

from nltk import tokenize

sid = vader.SentimentIntensityAnalyzer()
demo_tweet['sentiment_compound_polarity']=demo_tweet.text.apply(lambda x:sid.polarity_scores(x)['compound'])
demo_tweet['sentiment_neutral']=demo_tweet.text.apply(lambda x:sid.polarity_scores(x)['neu'])
demo_tweet['sentiment_negative']=demo_tweet.text.apply(lambda x:sid.polarity_scores(x)['neg'])
demo_tweet['sentiment_pos']=demo_tweet.text.apply(lambda x:sid.polarity_scores(x)['pos'])
demo_tweet['sentiment_type']=''
demo_tweet.loc[demo_tweet.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
demo_tweet.loc[demo_tweet.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
demo_tweet.loc[demo_tweet.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
demo_tweet.head()
demo_tweet.sentiment_type.value_counts().plot(kind='bar',title="Sentiment analysis")
## Sort the tweets in descending order based on nmber of retweets..

Famous_tweet=demo_tweet.sort_values('retweetCount', ascending = False).head()['text']
Famous_tweet.head()
demo_tweet['hour'] = pd.DatetimeIndex(demo_tweet['created']).hour
Retweets_hour = demo_tweet.groupby(['hour'])['retweetCount'].sum()
### find total tweets in hour..

demo_tweet['total'] = 1
Total_tweets_hour = demo_tweet.groupby(['hour'])['total'].sum()
#tweets_hour
Tweets_Hour=pd.DataFrame({'Rtweet_hr':Retweets_hour,'Ttl_twt_hr':Total_tweets_hour})
Tweets_Hour.plot.barh(stacked=True, colormap = 'viridis')
#Tweets_Hour.head(20)
## Not done yet..If you find answer share with me.
demo_tweet['statusSource_new'] = ''

for i in range(len(demo_tweet['statusSource'])):
    m = re.search('(?<=>)(.*)', demo_tweet['statusSource'][i])
    try:
        demo_tweet['statusSource_new'][i]=m.group(0)
    except AttributeError:
        demo_tweet['statusSource_new'][i]=demo_tweet['statusSource'][i]
        
print(demo_tweet['statusSource_new'].head())   
demo_tweet['statusSource_new2'] = ''

for i in range(len(demo_tweet['statusSource_new'])):
    if demo_tweet['statusSource_new'][i] not in ['Twitter for Android ','Twitter Web Client ','Twitter for iPhone ', 'Mobile Web (M5) ', 'Facebook']:
        demo_tweet['statusSource_new2'][i] = 'Others'
    else:
        demo_tweet['statusSource_new2'][i] = demo_tweet['statusSource_new'][i] 
#print(tweets['statusSource_new2'])       

tweets_by_type2 = demo_tweet.groupby(['statusSource_new2'])['retweetCount'].sum()
