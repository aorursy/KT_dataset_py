!pip install tweepy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tweepy as tw # To extarct the twitter data
from tqdm import tqdm
consumer_api_key = 'Type your API KEY here '
consumer_api_secret = 'Type your API KEY SECRET here'
auth = tw.OAuthHandler(consumer_api_key, consumer_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)
search_words = "#ipl2020 -filter:retweets" #Type you keywork here instead of #covidvaccine
#You can fix a time frame with the date since and date until parameters
date_since = "2020-08-17"
date_until="2020-08-20"
# Collect tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since,
              until=date_until     
              ).items(7500) #We instruct the cursor to return maximum of 7500 tweets
tweets_copy = []
for tweet in tqdm(tweets):
    tweets_copy.append(tweet)
print(f"New tweets retrieved: {len(tweets_copy)}")

for tweet in tqdm(tweets_copy):
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_created': tweet.user.created_at,
                                               'user_followers': tweet.user.followers_count,
                                               'user_friends': tweet.user.friends_count,
                                               'user_favourites': tweet.user.favourites_count,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': tweet.text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               'source': tweet.source,
                                               'is_retweet': tweet.retweeted}, index=[0]))
tweets_df
tweets_df.to_csv('ipl2020.csv',index=False)
## And Voila! You now have the tweets you required for the given time frame! Kindly upvote the kernel if you find it useful!