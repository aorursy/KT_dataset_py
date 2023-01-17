from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from datetime import datetime
from pytz import timezone

from types import FunctionType, MethodType
def refer_args(x):
    if type(x) is MethodType or type(x) is FunctionType:
        print(*x.__code__.co_varnames, sep='\n')
    else:
        print(*[x for x in dir(x) if not x.startswith('__')], sep='\n')

datetime.now(timezone('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')
!pip install GetOldTweets3
import os
from time import sleep

import pandas as pd
from tqdm.notebook import tqdm
import GetOldTweets3 as got
from GetOldTweets3.manager import TweetManager, TweetCriteria
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
sr_id_train = df_train['id']
sr_text_train = df_train['text']
sr_id_test = df_test['id']
sr_text_test = df_test['text']
def tweet2dict(id_, tweet):
    return {
        'id':id_,
        'permalink': tweet.permalink,
        'username': tweet.username,
        'favorites': tweet.favorites,
        'retweets': tweet.retweets,
        'text': tweet.text,
        'date': tweet.date,
        'mentions': tweet.mentions,
        'hashtags': tweet.hashtags,
        'geo': tweet.geo,
        'urls': tweet.urls,
    }

columns = [
    'id',
    'permalink',
    'username',
    'favorites',
    'retweets',
    'text',
    'date',
    'mentions',
    'hashtags',
    'geo',
    'urls',
]
dtype = {
    'id':'int',
    'favorites': 'int',
    'retweets': 'int',
}
df_tweet_train = pd.DataFrame(columns=columns)
df_tweet_train = df_tweet_train.astype(dtype)

idx = 0
id_, query = sr_id_train[idx], sr_text_train[idx]
tweetCriteria = TweetCriteria().setQuerySearch(query).setMaxTweets(10)
tweets = TweetManager.getTweets(tweetCriteria)
if tweets:
    tweet = min((tweet for tweet in tweets), key=lambda tweet:tweet.date)
    tweet_dict = tweet2dict(id_, tweet)
else:
    tweet_dict = {'id':id_}
df_tweet_train = df_tweet_train.append(tweet_dict, ignore_index=True)

id_, query
tweet_dict
file = 'tmp_file'
if os.path.isfile(file):
    df_tweet_train = pd.read_csv(file)
    idx = len(df_tweet_train)
else:
    df_tweet_train = pd.DataFrame(columns=columns)
    idx = 0
%%time
loop = tqdm(
    enumerate(zip(sr_id_train[idx:], sr_text_train[idx:]), idx+1),
    total=len(sr_id_train),
    initial=idx
)

for i, (id_, query) in loop:
    tweetCriteria = TweetCriteria().setQuerySearch(query).setMaxTweets(10)
    try:
        tweets = TweetManager.getTweets(tweetCriteria)
    except SystemExit:
        tweets = []
    if tweets:
        tweet = min((tweet for tweet in tweets), key=lambda tweet:tweet.date)
        tweet_dict = tweet2dict(id_, tweet)
    else:
        tweet_dict = {'id':id_}
    df_tweet_train = df_tweet_train.append(tweet_dict, ignore_index=True)
    sleep(2)
    if i % 500 == 0:
        df_tweet_train.to_csv(f'data/train_tweet_{i}.csv', index=False)
df_tweet_train = df_tweet_train.astype({'id':int})
df_tweet_train.to_csv('train_tweet.csv', index=False)
file = 'tmp_file'
if os.path.isfile(file):
    df_tweet_test = pd.read_csv(file)
    idx = len(df_tweet_test)
else:
    df_tweet_test = pd.DataFrame(columns=columns)
    idx = 0
%%time
loop = tqdm(
    enumerate(zip(sr_id_test[idx:], sr_text_test[idx:]), idx+1),
    total=len(sr_id_test),
    initial=idx
)

for i, (id_, query) in loop:
    tweetCriteria = TweetCriteria().setQuerySearch(query).setMaxTweets(10)
    try:
        tweets = TweetManager.getTweets(tweetCriteria)
    except SystemExit:
        tweets = []
    if tweets:
        tweet = min((tweet for tweet in tweets), key=lambda tweet:tweet.date)
        tweet_dict = tweet2dict(id_, tweet)
    else:
        tweet_dict = {'id':id_}
    df_tweet_test = df_tweet_test.append(tweet_dict, ignore_index=True)
    sleep(2)
    if i % 500 == 0:
        df_tweet_test.to_csv('test_tweet_backup.csv', index=False)
df_tweet_test = df_tweet_test.astype({'id':int})
df_tweet_test.to_csv('test_tweet.csv', index=False)