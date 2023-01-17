# If tweepy is not installed in your current environment
! pip install tweepy
import tweepy #The Twitter API
from time import sleep
from datetime import datetime
from textblob import TextBlob #For Sentiment Analysis
import numpy as np
import pandas as pd
import re
import warnings

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS

#nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

%matplotlib inline

# tweets = pd.read_csv('tweets_all.csv', encoding = "ISO-8859-1")
consumer_key = '6i7pvwjFLtXNcecug1xetUSqf'
consumer_secret = '5jGK2zlTgKnk90i5MXRcmpOpt1qvBWSsfsst6EqkO7VS1mPa1W'
access_token = '137590128-Qw9foqrswzhkV0yQGAVIFVOt7Y5EEW7k73SuyQWc'
access_token_secret = 'q47C3TZqmIbBg7TNd2oPDBcCE2d0SDPUdQaJDSsbGMbtq'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
search = api.search(q='#datascience', count=100) #**supply whatever query you want here**
searched_tweets = [each._json for each in search]
json_strings = [json.dumps(json_obj) for json_obj in searched_tweets]
searched_tweets[0]['user']
def flatten_tweets(tweets_json):
    """ Flattens out tweet dictionaries so relevant JSON
        is in a top-level dictionary."""
    tweets_list = []
    
    # Iterate through each tweet
    for tweet in tweets_json:
        tweet_obj = json.loads(tweet)
    
        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
            
        tweets_list.append(tweet_obj)
    return tweets_list

tweet= flatten_tweets(json_strings)
tweet_d = pd.DataFrame(tweet)
tweet_d.shape
tweets_with_quoted_status = tweet_d[~tweet_d.quoted_status.isnull()]['quoted_status'].reset_index(drop=True)
tweet_d.columns
tweets = pd.DataFrame()
msgs = []
msg =[]
search[0].user.followers_count
for tweet in search:
    msg = [tweet.text, tweet.source, tweet.source_url, tweet.user.location, tweet.user.followers_count] 
    msg = tuple(msg)                    
    msgs.append(msg)
    
tweets = pd.DataFrame(msgs)
tweets.head()
tweets.columns = ['text', 'source', 'url', 'location', 'followers_count']
tweets.head()
tweets['text'][1]
#Preprocessing delete "RT" and "@username":
tweets['tweetos'] = '' 

#add tweetos first part
for i in range(len(tweets['text'])):
    try:
        tweets['tweetos'][i] = tweets['text'].str.split(' ')[i][0]
    except AttributeError:    
        tweets['tweetos'][i] = 'other'

#Preprocessing tweetos. select tweetos contains 'RT @'
for i in range(len(tweets['text'])):
    if tweets['tweetos'].str.contains('@')[i]  == False:
        tweets['tweetos'][i] = 'other'
        
# remove URLs, RTs, and twitter handles
for i in range(len(tweets['text'])):
    tweets['text'][i] = " ".join([word for word in tweets['text'][i].split()
                                if 'http' not in word and '@' not in word and '<' not in word])


tweets['text'][1]
tweets['text'] = tweets['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
tweets['text'] = tweets['text'].apply(lambda x: re.sub('  ', ' ', x))
tweets['text'][1]
def wordcloud(tweets,col):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets[col]]))
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Good Morning Datascience+")
wordcloud(tweets,'text')
tweets['location'] = tweets['location'].apply(lambda x: ' ' if x==None else str(x))
wordcloud(tweets, 'location')
tweets['source'][69]
tweets['source_new'] = ''

for i in range(len(tweets['source'])):
    m = re.search('(?i)<a([^>]+)>(.+?)</a>', tweets['source'][i])
    try:
        tweets['source_new'][i]=m.group(0)
    except AttributeError:
        tweets['source_new'][i]=tweets['source'][i]
        
tweets['source_new'] = tweets['source_new'].str.replace('', ' ', case=False)
tweets_by_type = tweets.groupby(['source_new'])['followers_count'].sum()
plt.title('Number of followers by Source', bbox={'facecolor':'0.8', 'pad':0})
tweets_by_type.transpose().plot(kind='bar',figsize=(20, 10))
tweets['source_new2'] = ''

for i in range(len(tweets['source_new'])):
    if tweets['source_new'][i] not in ['Twitter for Android ','Instagram ','Twitter Web Client ','Twitter for iPhone ']:
        tweets['source_new2'][i] = 'Others'
    else:
        tweets['source_new2'][i] = tweets['source_new'][i] 

tweets_by_type2 = tweets.groupby(['source_new2'])['followers_count'].sum()
tweets_by_type2.rename("",inplace=True)
explode = (1,0,0,0,0)
tweets.groupby(['source_new2'])['followers_count'].count()
tweets['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in tweets['text']]       
vectorizer = TfidfVectorizer(max_df=0.5,max_features=10000,min_df=10,stop_words='english',use_idf=True)
X = vectorizer.fit_transform(tweets['text_lem'].str.upper())
sid = SentimentIntensityAnalyzer()
tweets['sentiment_compound_polarity']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.text_lem.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'
tweets_sentiment = tweets.groupby(['sentiment_type'])['sentiment_neutral'].count()
tweets_sentiment.rename("",inplace=True)
explode = (1, 0, 0)
plt.subplot(221)
tweets_sentiment.transpose().plot(kind='barh',figsize=(20, 20))
plt.title('Sentiment Analysis 1')
plt.subplot(222)
tweets_sentiment.plot(kind='pie',figsize=(20, 20),autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.)
plt.title('Sentiment Analysis 2')
plt.show()
tweets[tweets.sentiment_type == 'NEGATIVE'].text.reset_index(drop = True)[1]
sid.polarity_scores("rt will you be at mozfest this weekend don’t miss our interactive conversation about youthsurveillance with")
# Warning: only 1-3% of Twitter data have geographical data
import pycountry
country = {}
for i in list(pycountry.countries):
    country[i.alpha_2] = i.name
    
loc = []
for i in tweets["location"]:
    j = i.split(",")
    if len(j)==1:
        if j[0].strip() in country.keys():
            loc.append(country[j[0].strip()])
        elif j[0].strip() in country.values():
            loc.append(j[0])
    else:
        if j[len(j)-1].strip() in country.keys():
            loc.append(country[j[len(j)-1].strip()])
        elif j[len(j)-1].strip() in country.values():
            loc.append(j[len(j)-1])
            
for i in range(len(loc)):
    loc[i] = loc[i].strip()
    
loc = list(loc)
unique_loc = list(set(loc))

c = []
for i in unique_loc:
    c.append(loc.count(i))
    
q = pd.DataFrame()
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")
latitude = [] 
long = []
for i in unique_loc:
    if i != None:
        location = geolocator.geocode(i)
        if location!=None:
            latitude.append(location.latitude)#, location.longitude)
            long.append(location.longitude)
tweets['latitude'] = pd.Series(latitude)
tweets['longitude'] = pd.Series(long)
tweets.head()
q = pd.DataFrame({"latitude":latitude,"longitude":long,"location":unique_loc,"count":c})
import folium
m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
for i in range(0,len(q)):
    popup= folium.Popup(q.iloc[i]['location'], parse_html=True)
    folium.Marker([q.iloc[i]['latitude'], q.iloc[i]['longitude']], popup=popup).add_to(m)
m