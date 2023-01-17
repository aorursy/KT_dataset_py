import os 

import sys

import numpy as np

import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tweets = []



with open('../input/tweets_collector.txt', "r", encoding = "utf-8") as f :

    for tweet in f :

        tweets.append(tweet)
print(len(tweets))
tweets_from_text_to_csv = pd.DataFrame(tweets, columns = {'tweet_collector'})

print(tweets_from_text_to_csv)
tweets_from_text_to_csv.head(1)
tweets_from_text_to_csv.to_csv('tweets.csv')