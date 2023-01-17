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
import json

import pymongo

import tweepy



with open('consumer_key.txt', 'r') as f:

    consumer_key =  f.read()

f.closed



with open('consumer_secret.txt', 'r') as f:

    consumer_secret = f.read()

f.closed



with open('access_key.txt', 'r') as f:

    access_key = f.read()

f.closed



with open('access_secret.txt', 'r') as f:

     access_secret = f.read()

f.closed





#Authentication

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)





USER_NAME = "gatdelapena"

user = api.get_user(id=USER_NAME)

print('Username:',user.screen_name)

print('User Created at:',user.created_at)

timeline = api.user_timeline(id=USER_NAME,count=1)

#Retrieve a Status object...

twitter= []  

followers = tweepy.Cursor(api.followers, screen_name=USER_NAME).items()  



# print(followers)

try:

    for follower in followers:

        if follower.protected==False:

            new_tweets = api.user_timeline(screen_name = follower.screen_name,count=1, tweet_mode="extended")

            for tweet in new_tweets:

                twitter.append({'Screen Name':follower.screen_name,'Created at':follower.created_at, 'Tweet':tweet.full_text})

            

except tweepy.TweepError:

    print("Failed to run the command on that user, Skipping...")

        

try:

    #use your database name, user and password here:

    #mongodb://<dbuser>:<dbpassword>@<mlab_url>/<database_name>

    with open("credentials.txt", 'r') as f:

        [name,password,url]=f.read().splitlines()

        conn=pymongo.MongoClient("mongodb+srv://{}:{}@{}".format(name,password,url))

    print ("Connected successfully!!!")

except pymongo.errors.ConnectionFailure as e:

        print ("Could not connect to MongoDB: %s" % e)   



db = conn['twitter']

collection = db["info"]

db['info'].delete_many({})

db["info"].insert_many(twitter)

db["info"].count_documents({})