# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Set up input list

keywords = ['stomach ache','ebola','fever','vaccin','epidemic','#ebola','#Virus','vomit','virus ebola','symptoms ebola','"diarrhea due to ebola"']

#n = int(input("Enter the number of keywords : ")) 

#for i in range(0,n):

#    keywords.append(input())

print(keywords)
#Install and import module for this part

!pip install googletrans

from googletrans import Translator
#Translate

translator = Translator()

keywords_trad = []

translations = translator.translate(keywords, dest='fr')

for translation in translations:

    keywords_trad.append(translation.text)

    print(translation.origin, ' -> ', translation.text)
#Install and import module for this part

!pip install twython

from twython import Twython

import json
# Load credentials from json file

with open("/kaggle/input/credentials/twitter_credentials.json", "r") as file:

    creds = json.load(file)
# Instantiate an object

python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])



# Create our query and search tweets

dict_ = {'user': [], 'date': [], 'text': [], 'location': []}

for i in keywords_trad:

    query = {'q': ' {}  -filter:retweets'.format(i),

             #'geocode':'-4.437584, 15.252361,1000km',

            'result_type': 'mixed',

            'count': 1000,

            'lang': 'fr',

            }

    for status in python_tweets.search(**query)['statuses']:

        dict_['user'].append(status['user']['screen_name'])

        dict_['location'].append(status['user']['location'])

        dict_['date'].append(status['created_at'])

        dict_['text'].append(status['text'])
#Modules

import pandas as pd

from datetime import datetime 

from email.utils import mktime_tz, parsedate_tz



#Function to convert date format

def parse_datetime(value):

    time_tuple = parsedate_tz(value)

    timestamp = mktime_tz(time_tuple)

    return datetime.fromtimestamp(timestamp)



# Structure data in a pandas DataFrame for easier manipulation

df = pd.DataFrame(dict_)

#convert twitter date to another date time format

for i in range(len(df['date'])):

    df['date'][i] = parse_datetime(df['date'][i])

df['date'] = df['date'].map(lambda x: str(x)[0:10])

df

#If we want to save this dataframe

#df.to_csv('tweets_saved.csv')
#Load an older dataframe with older tweets

older_tweets = pd.read_csv("/kaggle/input/oldertweets/tweets2.csv")

older_tweets = older_tweets.drop(older_tweets.columns[[0]], axis=1)

older_tweets['date'] = older_tweets['date'].map(lambda x: str(x)[0:10])

older_tweets
#Merge dataframes into a big one and sort it by date

data = [df,older_tweets]

tweets = pd.concat(data)

tweets = tweets.drop_duplicates()

tweets = tweets.sort_values(by='date')

tweets
#Module for regular expression matching operations

import re

#Function that found our word in tweets

def word_in_text(word, text):

    word = word.lower()

    text = text.lower()

    match = re.search(word, text)

    if match:

        return 1

    return 0

#Function to delete emojis

def deEmojify(inputString):

    return inputString.encode('ascii', 'ignore').decode('ascii')



data_kw = pd.DataFrame()

for k in keywords_trad:

    data_kw[k] = tweets['text'].apply(lambda tweet: word_in_text(k, tweet))

data_kw['date'] = tweets['date']

data_kw_date = data_kw.groupby('date').sum()

data_kw_date
#Modules

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import ticker, cm



#Count and time series

count = data_kw_date.cumsum()

time = ['day'+'{}'.format(i+1) for i in range(len(data_kw_date.index))]



#Plot

plt.figure(figsize = (7,5))

plt.style.use('seaborn-paper')

col=iter(cm.rainbow(np.linspace(0,1,len(data_kw_date.columns))))

for k in list(data_kw_date.columns):

    c = next(col)

    plt.plot(time,count[k],'o-',color=c, label='{}'.format(k))

plt.xlabel("Time", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.tick_params(axis = 'both', labelsize = 12)

plt.legend(fontsize=12)

plt.show()

#Find which keywords are more used and more relevent for our reasearch

data_count = pd.DataFrame({'Count': data_kw_date.sum(axis = 0, skipna = True)}).sort_values(by=['Count'])

data_count.reset_index(level=0, inplace=True)

data_count = data_count.rename(columns={"index": "Keywords"})



#Bar chart

plt.figure(figsize=(10, 10))

data_count.plot.barh(x='Keywords',y='Count',color='deepskyblue')

plt.xlabel('Occurence', fontsize=15)

plt.ylabel('keywords', fontsize=15)

plt.legend(fontsize=12)

plt.tick_params(axis = 'both', labelsize = 12)

plt.title("Count of keywords founded over {} Tweets".format(len(tweets['text'])), fontsize = 20)

plt.show()
!pip install geopy

!pip install gmplot

from gmplot import gmplot

from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter
data_kw['location'] = tweets['location']

data_kw_location = data_kw.groupby('location').sum()

for i in range(len(data_kw_location.index.values)):

    data_kw_location.index.values[i] = deEmojify(data_kw_location.index.values[i])

data_kw_location

geolocator = Nominatim(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36")

# Go through all tweets and add locations to 'coordinates' dictionary

coordinates = {'latitude': [], 'longitude': []}

#Tracking Ebola keyword

ebola = data_kw_location['ebola']

ebola = ebola[ebola.values > 0]

for count,loc in enumerate(ebola.index):

    try:

        location = geolocator.geocode(loc)

        # If coordinates are found for location

        if location:

            coordinates['latitude'].append(location.latitude)

            coordinates['longitude'].append(location.longitude)

            

    # If too many connection requests

    except:

        pass
# Instantiate and center a GoogleMapPlotter object to show our map

gmap = gmplot.GoogleMapPlotter(30, 0, 3)

gmap.heatmap(coordinates['latitude'], coordinates['longitude'], radius=20)

gmap.draw("map_ebola.html")
#Heatmap of regon where the keyword 'ebola' was used

from IPython.core.display import Image

Image("../input/map-image-ebola/map.jpg")