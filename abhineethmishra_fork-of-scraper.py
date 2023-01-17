# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install tweepy
import tweepy

import pandas as pd 

import csv





tsv_file='../input/t4sa_text_sentiment.tsv'

csv_table=pd.read_table(tsv_file,sep='\t')

df = csv_table



tweets = []



consumer_key = 'ZulxjeGQgIyeSBtPDvOzxEGX1'

consumer_secret = 'SgRwQPdfNsbgCeXAK6uVaCVe5PzfFsdi9VGCPa3jnuFA37Pd4C'

access_token = '1112962597236563969-54YP8knUJHo1dqfRu015iWuq2X2gjw'

access_token_secret = 'bn04LVrAlVsyVTLziM0M9OLVlkKDfUKKrweQu1l6ygq8A'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)



for j in range(500,1179):

    ids=list(df.TWID.iloc[j*1000:(j+1)*1000])

    try:

        for i in range((len(ids) // 100)):

            end_loc = min((i + 1) * 100, len(ids))

            results = api.statuses_lookup(id_=ids[i*100:end_loc]) 

            tweets.extend([(result.id_str, result.text) for result in results])

    except tweepy.TweepError:

            print (tweepy.TweepError.errormessage)

    print('Done {}'.format((j+1)*1000))



print(len(tweets))
res=pd.DataFrame(tweets)

res.to_csv('scrapedTweetsFrom500.csv', sep='\t')