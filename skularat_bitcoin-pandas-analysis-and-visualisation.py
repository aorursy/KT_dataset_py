#Setup the Environment by importing the libraries needed for this notebook

import sys
import os
import pandas as pd
import cufflinks as cf
import plotly as py
import plotly.graph_objs as go
import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot #allow offline and #notebook 

init_notebook_mode(connected=True) #allow plotly graphs to work inside notebook
column_name=['created_at','text','screen_name','followers_count','favourite_count','hashtaglist','device_used','feelings']
bitcoin_tweets='../input/bitcointweets.csv'
tweet=pd.read_csv(bitcoin_tweets,parse_dates=['created_at'],encoding='utf-8',names=column_name) #parse dates
pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr', False)
tweet.head()
tweet=tweet[~tweet['text'].str.contains('RT')] #remove retweets
tweet.head()
tweet['hashtaglist']=tweet['hashtaglist'].str.strip('[]') #cleanup square brackets
tweet['feelings']=tweet['feelings'].str.strip('[]')
tweet.head() #data looks ok now. we can ignore the utf characters
feelings=tweet.groupby(['feelings'])['feelings'].count().reset_index(name='feelings_count')
feelings.head()
#feelings.iplot(kind='bar',color="blue")
iplot([go.Bar(x=feelings['feelings'],y=feelings['feelings_count'])])
utc_offset = +8 #singapore time 
tweet['Local_Time']=tweet.created_at + pd.to_timedelta(utc_offset,unit='h')
tweet.head() # checking our new column 
tweet['hour']=[time.hour for time in tweet.Local_Time]
tweet.head() #great we have the hour column
tweets_per_hour=tweet.groupby(['hour'])['text'].count().reset_index(name='tweets_count') # grouped to get number of tweets from author per hour
#tweets_per_hour.iplot(kind='scatter',x='hour',y='tweets_count',title="Tweets Per Hour",xTitle="Hour",yTitle="Tweet Count",color='green')
iplot([go.Scatter(x=tweets_per_hour['hour'],y=tweets_per_hour['tweets_count'])])
tweet['mention']=tweet['text'].str.startswith('@') #returns a boolean for all that match and adds new column
tweet.head() # notice new column 'mention'
# check if mention is true
mentions=tweet[tweet['mention']==True].groupby(['screen_name','mention'])['mention'].count().sort_values(ascending=False).reset_index(name='mentions').head(20)
#mentions.iplot(kind='bar',x='screen_name',y='mentions',color='darkblue',xTitle='Tweet Account',yTitle="Mentions Count",title="Top 20 Tweeters that have mentioned")
iplot([go.Bar(x=mentions['screen_name'],y=mentions['mentions'])])
followers=tweet[['screen_name','followers_count']].sort_values(by='followers_count',ascending=False).drop_duplicates('screen_name').head(20)
followers
#followers.iplot(kind='bar',x='screen_name',xTitle='Tweet Account',yTitle='Number Of Followers', title="Top 20 With Most Followers",color='purple')
iplot([go.Bar(x=followers['screen_name'],y=followers['followers_count'])])

tweet_per_account=tweet.groupby(['screen_name','hour'])['text'].count().sort_values(ascending=False).reset_index(name='num_of_tweets').head(20)
tweet_per_account.head()
accounts=list(set(tweet_per_account.screen_name))
traces=[]
for account in accounts:
    df=tweet_per_account[tweet_per_account['screen_name'].isin([account])]
    traces.append(go.Bar(x=df['hour'],y=df.num_of_tweets,name=account))
iplot({'data': traces,'layout':go.Layout(title='Tweets per account',barmode='grouped',xaxis={'tickangle': 30},margin={'b': 100})})