from warnings import filterwarnings

filterwarnings('ignore')
 !pip install tweepy
import tweepy, codecs







consumer_key = '5ecrgpF8KDhSZlvM8o0xSASVh'

consumer_secret = '5MdKdPRMpgGylXBZUlHXPwnxsgwjczusIeYw1JMTUUGQ0Dcdpi'

access_token = '330297491-56WABQRvTJnWTLG7bMnJT27Jlt9T5TYwUPiTR7hh'

access_token_secret = 'aCmW6YbiBjIycroI5zmoXGgpsnN46P5rXbCfx69KmRBMt'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)
# we can tweet with this code

# api.update_status('hello from python')
# we can see our friends on twitter with this code

#api.friends()
fk = api.me()
# you can follow me :)

fk.screen_name
fk.followers_count
#fk.friends
for friend in fk.friends(count=10):

    print(friend.screen_name)
dir(fk)
# she is my sister

user = api.get_user(id = 'bsrakrsn')
user.screen_name
user.followers_count
user.profile_image_url
public_tweets = api.home_timeline(count=10)
for tweet in public_tweets:

    print(tweet.text)
name = 'AndrewYNg'

tweet_count = 10



user_timeline = api.user_timeline(id = name, count=tweet_count)



for i in user_timeline:

    print(i.text)
retweets = api.retweets_of_me(count=10)

for retweet in retweets:

    print(retweet.text)
retweets
results = api.search(q = '#datascience',

                    lang = 'tr',

                    result_type = 'recent',

                    count = 1000000 )
import pandas as pd
def tweets_df(results):

    id_list = [tweet.id for tweet in results]

    data_set = pd.DataFrame(id_list,columns=['id'])

    

    

    data_set['text'] = [tweet.text for tweet in results]

    data_set['created_at'] = [tweet.created_at for tweet in results]

    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]

    data_set['name'] = [tweet.author.name for tweet in results]

    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]

    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]

    data_set['user_location'] = [tweet.author.location for tweet in results]

    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]

    

    return data_set
data = tweets_df(results)
data.head()
AndrewNg = api.get_user('AndrewYNg')
AndrewNg.name
AndrewNg.id
AndrewNg.url
AndrewNg.verified
AndrewNg.screen_name
AndrewNg.statuses_count
AndrewNg.favourites_count
AndrewNg.friends_count
tweets = api.user_timeline(id = 'AndrewYNg')
"""for i in tweets:

    print(i.text)"""
def timeline_df(tweets):

    id_list = [tweet.id for tweet in tweets]

    data_set = pd.DataFrame(id_list,columns=['id'])

    

    

    data_set['text'] = [tweet.text for tweet in tweets]

    data_set['created_at'] = [tweet.created_at for tweet in tweets]

    data_set['retweet_count'] = [tweet.retweet_count for tweet in tweets]

    data_set['favorite_count'] = [tweet.favorite_count for tweet in tweets]

    data_set['source'] = [tweet.source for tweet in tweets]



    

    return data_set
timeline_df(tweets)
def timeline_df(tweets):

    df = pd.DataFrame()

    

    df['id'] = list(map(lambda tweet:tweet.id, tweets))

    df['created_at'] = list(map(lambda tweet:tweet.created_at, tweets))

    df['text'] = list(map(lambda tweet:tweet.text, tweets)) 

    df['favorite_count'] = list(map(lambda tweet:tweet.favorite_count, tweets))

    df['retweeted_count'] = list(map(lambda tweet:tweet.retweet_count, tweets))

    df['source'] = list(map(lambda tweet:tweet.source, tweets))

    return df
tweets = api.user_timeline(id = 'AndrewYNg',count=10000)
df = timeline_df(tweets)
df.info()
df.sort_values('retweeted_count', ascending= False)
df.sort_values('favorite_count', ascending= False)[['text', 'favorite_count']].iloc[0:3]
df.sort_values('favorite_count', ascending= False)['text'].iloc[0]
df.head()
%config InlineBacend.figure_format = 'retina'

import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df.favorite_count, kde=False ,color='blue')

plt.xlim(-100,15000)
plt.figure(figsize=(10,6))

sns.distplot(df.retweeted_count, color='red')

plt.xlim(-100,5000)
df['favorite_count'].mean()
df['favorite_count'].std()
df.head()
df['tweet_hour'] = df['created_at'].apply(lambda x: x.strftime('%H'))
df.head()
df['tweet_hour'] = pd.to_numeric(df['tweet_hour'])
df.info()
plt.figure(figsize=(10,6))

sns.distplot(df['tweet_hour'], kde=True, color='blue')
df['days'] = df['created_at'].dt.weekday_name
df.head()
gun_freq = df.groupby('days').count()['id']
gun_freq.plot.bar(x='days', y='id')
source_freq = df.groupby('source').count()['id']
source_freq.plot.bar(x='source', y='id')
df.groupby('source').count()['id']
df.groupby(['source', 'tweet_hour','days'])[['tweet_hour']].count()
user = api.get_user(id = 'AndrewYNg', count= 10000)
friends = user.friends()

followers = user.followers()
def followers_df(follower):

    idler = [i.id for i in follower]

    df = pd.DataFrame(idler, columns=['id'])

    

    

    df['created_at'] = [i.created_at for i in follower]

    df['screen_name'] = [i.screen_name for i in follower]

    df['location'] = [i.location for i in follower]

    df['followers_count'] = [i.followers_count for i in follower]

    df['statuses_count'] = [i.statuses_count for i in follower]

    df['friends_count'] = [i.friends_count for i in follower]

    df['favourites_count'] = [i.favourites_count for i in follower]

    

    return df
df = followers_df(followers)
df.head()
df.info()
df.index = df['screen_name']
s_data = df[['followers_count', 'statuses_count']]
s_data
s_data['followers_count'] = s_data['followers_count'] +0.01
s_data['statuses_count'] = s_data['statuses_count'] +0.01
s_data
s_data = s_data.apply(lambda x:(x-min(x)) / (max(x)- min(x))) #doing standardization
s_data['followers_count'] = s_data['followers_count'] +0.01

s_data['statuses_count'] = s_data['statuses_count'] +0.01
s_data.head()
score = s_data['followers_count'] * s_data['statuses_count']
score
score.sort_values(ascending = False)
score[score>score.median() + score.std()/2].sort_values(ascending=False)
score.median()
s_data['score'] =score
import numpy as np
s_data['segment'] = np.where(s_data['score'] >=score.median() + score.std()/len(score) , 'A', 'B')
s_data
a = api.user_timeline(id= 'AndrewYNg',count=5)
for i in a:

    print(i.text)
def country_codes():

    places = api.trends_available()

    all_woeids = {place['name'].lower(): place['woeid'] for place in places}

    return all_woeids
# country_codes()
def country_woeid(country_name):

    country_name = country_name.lower()

    trends = api.trends_available()

    all_woeids = country_codes()

    return all_woeids[country_name]
country_woeid('turkey')
trends = api.trends_place(id= 23424969 )
import json

#print(json.dumps(trends, indent=1))
turkey = api.trends_place(id= 23424969 )

trends = turkey[0]['trends']
tweets = api.search(q= '#datascience', lang='en',

                     result_type='recent', counts = 1000)
def hashtag_df(results):

    id_list = [tweet.id for tweet in results]

    data_set = pd.DataFrame(id_list,columns=['id'])

    

    

    data_set['text'] = [tweet.text for tweet in results]

    data_set['created_at'] = [tweet.created_at for tweet in results]

    data_set['retweeted'] = [tweet.retweeted for tweet in results]

    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]

    data_set['name'] = [tweet.author.name for tweet in results]

    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]

    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]

    data_set['user_location'] = [tweet.author.location for tweet in results]

    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]

    

    return data_set
df = hashtag_df(tweets)
df.shape
df
df['tweet_hour'] = df['created_at'].apply(lambda x: x.strftime('%H'))
df['tweet_hour'] = pd.to_numeric(df['tweet_hour'])
plt.figure(figsize=(10,6))

sns.distplot(df['tweet_hour'], kde=True, color='blue')
df['days'] = df['created_at'].dt.weekday_name
gun_freq = df.groupby('days').count()['id']
gun_freq.plot.bar(x='days', y='id')


df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))



df['text'] = df['text'].str.replace('[^\w\s]', '')



df['text'] = df['text'].str.replace('[\d]','')





import nltk

from nltk.corpus import stopwords

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))



#lemmi

from textblob import Word

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))



df['text'] = df['text'].str.replace('rt', '')
df.text
freq_df = df['text'].apply(lambda x:pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
freq_df.columns = ['words', 'freqs']
freq_df.sort_values('freqs',ascending=False)
freq_df.shape
a = freq_df[freq_df.freqs > freq_df.freqs.mean() + 

       freq_df.freqs.std()] # this code for the being more meaningful
a.plot.bar(x= 'words', y= 'freqs')
import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud , STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
text = " ".join(i for i in df.text)
text
wc = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(10,6))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.tight_layout(pad = 0)

plt.show()
df
from textblob import TextBlob
def sentiment_score(df):

    text = df['text']

    

    for i in range(0, len(text)):

        textB = TextBlob(text[i])

        sentiment_score = textB.sentiment.polarity

        df.set_value(i, 'sentiment_score', sentiment_score)

        

        

        if sentiment_score < 0.00:

            sentiment_class = 'Negative'

            df.set_value(i, 'sentiment_class', sentiment_class)

            

        elif sentiment_score > 0.00:

            sentiment_class ='Positive'

            df.set_value(i, 'sentiment_class', sentiment_class)

        else:

            sentiment_class = 'Notr'

            df.set_value(i, 'sentiment_class', sentiment_class)

    return df



sentiment_score(df)
df.groupby('sentiment_class').count()['id']
sentiment_freq = df.groupby('sentiment_class').count()['id']
sentiment_freq.plot.bar(x= 'sentiment_class', y='id')
tweets = api.search(q = '#apple', lang='en', count=5000)
def hashtag_df(results):

    id_list = [tweet.id for tweet in results]

    data_set = pd.DataFrame(id_list,columns=['id'])

    

    

    data_set['text'] = [tweet.text for tweet in results]

    data_set['created_at'] = [tweet.created_at for tweet in results]

    data_set['retweeted'] = [tweet.retweeted for tweet in results]

    data_set['retweet_count'] = [tweet.retweet_count for tweet in results]

    data_set['name'] = [tweet.author.name for tweet in results]

    data_set['user_screen_name'] = [tweet.author.screen_name for tweet in results]

    data_set['user_followers_count'] = [tweet.author.followers_count for tweet in results]

    data_set['user_location'] = [tweet.author.location for tweet in results]

    data_set['Hashtags'] = [tweet.entities.get('hashtags') for tweet in results]

    

    return data_set
df = hashtag_df(tweets)
df.shape


df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))



df['text'] = df['text'].str.replace('[^\w\s]', '')



df['text'] = df['text'].str.replace('[\d]','')





import nltk

from nltk.corpus import stopwords

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))



#lemmi

from textblob import Word

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()] ))



df['text'] = df['text'].str.replace('rt', '')
def sentiment_score(df):

    text = df['text']

    

    for i in range(0, len(text)):

        textB = TextBlob(text[i])

        sentiment_score = textB.sentiment.polarity

        df.set_value(i, 'sentiment_score', sentiment_score)

        

        

        if sentiment_score < 0.00:

            sentiment_class = 'Negative'

            df.set_value(i, 'sentiment_class', sentiment_class)

            

        elif sentiment_score > 0.00:

            sentiment_class ='Positive'

            df.set_value(i, 'sentiment_class', sentiment_class)

        else:

            sentiment_class = 'Notr'

            df.set_value(i, 'sentiment_class', sentiment_class)

    return df
df = sentiment_score(df)
sentiment_freq = df.groupby('sentiment_class').count()['id']
sentiment_freq
sentiment_freq.plot.bar(x = 'sentiment_class', y= 'id')