import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading json module

import json



# Load WW_trends and US_trends data into the the given variables respectively

WW_trends = json.loads(open('../input/WWTrends.json').read())

US_trends = json.loads(open('../input/USTrends.json').read())



# Inspecting data by printing out WW_trends and US_trends variables

print(WW_trends, US_trends)
# Pretty-printing the results. First WW and then US trends.



print("WW trends:")

print(json.dumps(WW_trends[:5], indent=1))



print("\n", "US trends:")

print(json.dumps(US_trends[:5], indent=1))
# Extracting all the WW trend names from WW_trends

world_trends = set([trend['name']

                    for trend in WW_trends[0]['trends']])



# Extracting all the US trend names from US_trends

us_trends = set([trend['name']

                 for trend in US_trends[0]['trends']]) 



# Getting the intersection of the two sets of trends

common_trends = world_trends.intersection(us_trends)



# Inspecting the data

print(world_trends, "\n")

print(us_trends, "\n")

print (len(common_trends), "common trends:", common_trends)
# Loading the data

tweets = json.loads(open('../input/WeLoveTheEarth.json').read())



# Inspecting some tweets

tweets[0:2]
# Extracting the text of all the tweets from the tweet object

texts = [tweet['text'] for tweet in tweets]



# Extracting screen names of users tweeting about #WeLoveTheEarth

names = [name['screen_name'] for tweet in tweets for name in tweet['entities']['user_mentions']]



# Extracting all the hashtags being used when talking about this topic

hashtags = [hash['text'] for tweet in tweets for hash in tweet['entities']['hashtags']]



# Inspecting the first 10 results

print (json.dumps(texts[0:10], indent=1),"\n")

print (json.dumps(names[0:10], indent=1),"\n")

print (json.dumps(hashtags[0:10], indent=1),"\n")
# Importing modules

from collections import Counter



# Counting occcurrences/ getting frequency dist of all names and hashtags

for item in [names, hashtags]:

    c = Counter(item)

    # Inspecting the 10 most common items in c

    print (c.most_common(10), "\n")
# Extracting useful information from retweets

retweets = [(tweet['retweet_count'],

             tweet['retweeted_status']['favorite_count'], 

             tweet['retweeted_status']['user']['followers_count'], 

             tweet['retweeted_status']['user']['screen_name'], 

             tweet['text']) for tweet in tweets if 'retweeted_status' in tweet]
# Importing modules

import matplotlib.pyplot as plt

import pandas as pd



# Create a DataFrame and visualize the data in a pretty and insightful format

df = pd.DataFrame(retweets, 

                  columns=['Retweets','Favorites', 'Followers', 'ScreenName', 'Text']

                 ).groupby(['ScreenName','Text','Followers']).sum().sort_values(by=['Followers'], ascending=False)



df.style.background_gradient()
# Extracting language for each tweet and appending it to the list of languages

tweets_languages = []

for tweet in tweets: 

    tweets_languages.append(tweet['lang'])



# Plotting the distribution of languages

%matplotlib inline

plt.hist(tweets_languages)
# Congratulations!

print("High Five!!!")