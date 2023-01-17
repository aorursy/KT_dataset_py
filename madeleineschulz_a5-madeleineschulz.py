# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#As I mentioned in the MP1 assignment, I did these assignments out of order, so this is actually the data

# I will be using for my mini project.

#I connected to the Twitter API to examine what topics are trending in the Seattle area. This is interesting to me

# as a Twitter user and Seattle local.
#I looked at this page for reference on how to access the Twitter API: http://socialmedia-class.org/twittertutorial.html



import json

import tweepy



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

CONSUMER_KEY = user_secrets.get_secret("TwitterKey")

CONSUMER_SECRET = user_secrets.get_secret("TwitterSecretKey")

ACCESS_SECRET  = user_secrets.get_secret("TwitterSecretToken")

ACCESS_TOKEN = user_secrets.get_secret("TwitterToken")





auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)



api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)



#this prints the tweets on my home page:



for status in tweepy.Cursor(api.home_timeline).items(200):

    print(status._json)

#this prints the trending topics for a location based on the WOEID:



sea_trends = api.trends_place(id = 2490383)

print(json.dumps(sea_trends[0]["trends"]))
with open('sea_trends0217.txt', 'w') as outfile:

    json.dump(sea_trends[0]["trends"], outfile, indent=4)

    

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#this is as far as I could get with making a data frame out of the above trend data. 

# it seems like it's reading my data as a list, so it can't parse it? I don't know how to fix it.





import matplotlib.pyplot as plt

import pandas as pd



df = pd.read_json("/kaggle/working/sea_trends.txt")
df.sort_values('tweet_volume',ascending = False)
df = df[df['tweet_volume'].notna()]
print(df)
df.plot()
df.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false")