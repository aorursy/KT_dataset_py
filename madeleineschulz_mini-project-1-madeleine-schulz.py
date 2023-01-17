# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





        

import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.


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



for status in tweepy.Cursor(api.home_timeline).items(500):

    print(json.dumps(status._json, indent=4))

    

with open('homepage0225.txt', 'w') as outfile:

    json.dump(status._json, outfile, indent=4)
sea_trends = api.trends_place(id = 2490383)

print(json.dumps(sea_trends, indent=4))
#Now I'm just calling up each tweet trend individually, rather than having them all nested under the "trends" header.



sea_trends = api.trends_place(id = 2490383)

print(json.dumps(sea_trends[0]["trends"], indent = 4))
with open('sea_trends0225.txt', 'w') as outfile:

    json.dump(sea_trends[0]["trends"], outfile, indent=4)

    

#The below code prints the file name for my new file, as well as a couple files I created earlier this week with trends

#from 2/17 and 2/23. These dates are a bit arbitrary, but it's what I've got!

    

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt

import pandas as pd





df17 = pd.read_json("/kaggle/input/sea_trends0217.txt")

df23 = pd.read_json("/kaggle/input/sea_trends0223.txt")

df25 = pd.read_json("/kaggle/working/sea_trends0225.txt")
df17 = df17[df17['tweet_volume'].notna()]

df23 = df23[df23['tweet_volume'].notna()]

df17.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", color = "red",figsize=(20,10))

df23.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", color = "blue",figsize=(20,10))

df25.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", color = "green", figsize=(20,10))
df17 = df17[df17['tweet_volume'].notna()]

df23 = df23[df23['tweet_volume'].notna()]



ax = df25.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", color = "mediumspringgreen",

               label = "Number of Tweets 2/25", fontsize= 22, figsize=(20,15))



ax2 = df23.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", color = "seagreen", figsize=(20,15), 

          label = "Number of Tweets 2/23", fontsize = 22, ax=ax)



df17.plot(kind="bar", x = "name", y = "tweet_volume", use_index="false", 

          label = "Number of Tweets 2/17", color = "darkgreen",fontsize = 22, figsize=(20,15), ax=ax2)
homepage = pd.read_json("/kaggle/working/homepage0225.txt")

homepage