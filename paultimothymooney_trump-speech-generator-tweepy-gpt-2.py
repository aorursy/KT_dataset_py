!pip install tweepy
from kaggle_secrets import UserSecretsClient

ACCESS_TOKEN = UserSecretsClient().get_secret('ACCESS_TOKEN')

ACCESS_SECRET = UserSecretsClient().get_secret('ACCESS_SECRET')

CONSUMER_KEY = UserSecretsClient().get_secret('CONSUMER_KEY')

CONSUMER_SECRET = UserSecretsClient().get_secret('CONSUMER_SECRET')
import os

!git clone https://github.com/graykode/gpt-2-Pytorch.git

os.chdir('./gpt-2-Pytorch')

!curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

!pip install -r requirements.txt
import tweepy

import pandas as pd



def connect_to_twitter_OAuth():

    """adapted from https://towardsdatascience.com/my-first-twitter-app-1115a327349e"""

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth)

    return api



def extract_tweets(tweet_object):

    """adapted from https://towardsdatascience.com/my-first-twitter-app-1115a327349e"""

    tweet_list =[]

    for tweet in tweet_object:

        text = tweet.text

        created_at = tweet.created_at

        tweet_list.append({'created_at':created_at,'text':text,})

    df = pd.DataFrame(tweet_list, columns=[ 'created_at','text',])

    return df
api = connect_to_twitter_OAuth()

trump_tweets = api.user_timeline('realdonaldtrump')

df = extract_tweets(trump_tweets)

pd.set_option('display.max_colwidth', -1)

df.tail(10)
!python main.py --text "To the Democrats, Impeachment is partisan politics dressed up as principle.  This Scam going on right now by the Democrats against the Republican Party, and me, was all about a perfect phone call."
!rm -r /kaggle/working/*