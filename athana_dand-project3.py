# Importing all required libraries at once

import pandas as pd

# import tweepy

import os

import requests

import matplotlib.pyplot as plt

import glob

import numpy as np

import json

import time

import datetime

from scipy import stats

import statistics

import os

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import CSV for WeRateDogs Twitter archive

dfTwitter_archive = pd.read_csv('/kaggle/input/twitter/twitter-archive-enhanced.csv')
# we will skip the programmatic download here

'''

# programmatic download through Requests library

url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

directory = '/Users/nicolas/Google Drive/udacity/project 3'

response = requests.get(url)



with open(os.path.join(directory, url.split('/')[-1]), mode='wb') as file:

    file.write(response.content)

    

'''



dfImage_predictions = pd.read_csv('/kaggle/input/twitter/image-predictions.tsv', sep='\t')
# The original task included a programmatic download of data through the Twitter API. 

# Since I don't want to include the credentials here, I have commented it out and simply uploaded the final JSON file.

'''

# define credentials to call Twitter API

consumer_key = 'SECRET'

consumer_secret = 'SECRET'

access_token = 'SECRET'

access_secret = 'SECRET'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_secret)



api = tweepy.API(auth)





tweetIDs = dfTwitter_archive.tweet_id.to_list()

tweetErrors = []

tweetList = []



# loop for downloading tweet data through Twitter API

for tweet_id in tweetIDs:

    try:

        tweet = api.get_status(tweet_id, tweet_mode='extended')

        tweetList.append(tweet._json)

    except tweepy.TweepError as er:

        errors = {'ID':tweet_id, 'Error':er}

        tweetErrors.append(errors)

        pass



# converting dictionary into DataFrame

dfErrors = pd.DataFrame(tweetErrors, columns = ['ID','Error'])



# writing the data into the 'tweet_json.txt' file

file = 'tweet_json.txt'



with open(file, 'w') as outfile:

    for tweet in tweetList:

        json.dump(tweet, outfile)

        outfile.write('\n')'''
# reading in the json file and importing it into a proper DataFrame

jsonTwitter = []



file = '/kaggle/input/twitter/tweet_json.txt'



# opening the downloaded file

with open(file, 'r') as json_file:

    # running a loop for each line in the json file

    ln = json_file.readline()

    while ln:

        data = json.loads(ln)

        t_id = data['id']

        f_count = data['favorite_count']

        rt_count = data['retweet_count']

        #storing extracted information in dictionary

        json_data = {'id':t_id, 'favorites':f_count, 'retweets':rt_count}

        jsonTwitter.append(json_data)

        #reading new line

        ln = json_file.readline()



# converting dictionary into DataFrame

dfTwitter_api = pd.DataFrame(jsonTwitter, columns=['id', 'favorites', 'retweets'])



dfTwitter_api.head()
# starting with visual assessment first to get familiar with the DataFrame

dfTwitter_archive.head()
dfTwitter_archive.tail()
dfTwitter_archive.info()
dfTwitter_archive.describe()
# checking for any duplicates for 'tweet_id' in the DataFrame

dfTwitter_archive.tweet_id.nunique() == len(dfTwitter_archive)
# how many different sources do we have

dfTwitter_archive['source'].unique()
# how many different values do we have for the rating_numerator

dfTwitter_archive.rating_numerator.unique()
# looking up all 'unrealistic' values from 15 on

dfTwitter_archive.query('rating_numerator > 15').sort_values(by='rating_numerator', ascending=False)[['text', 'rating_numerator', 'rating_denominator']]
# how many different values do we have for the rating_denominator

dfTwitter_archive.rating_denominator.unique()
# looking up all 'unrealistic' values from 10 on

dfTwitter_archive.query('rating_denominator > 10').sort_values(by='rating_denominator', ascending=False)[['text', 'rating_numerator', 'rating_denominator']]
# searching for valid ratings from the original WeRateDogs Twitter profile

len(dfTwitter_archive[dfTwitter_archive['expanded_urls'].isnull()])
# if it's a real tweet, 'in_reply_to_status_id' and 'retweeted_status_id' need to be empty as well

dfTwitter_archive[dfTwitter_archive['expanded_urls'].isnull() & dfTwitter_archive['in_reply_to_status_id'].isnull() & dfTwitter_archive['retweeted_status_id'].isnull()]
# check if consistency existings within the stages column

dfTwitter_archive.doggo.unique(), \

dfTwitter_archive.floofer.unique(), \

dfTwitter_archive.pupper.unique(), \

dfTwitter_archive.puppo.unique()
# starting with visual assessment first to get familiar with the DataFrame

dfImage_predictions.head()
dfImage_predictions.tail()
dfImage_predictions.info()
dfImage_predictions.describe()
# checking for duplicate entries via 'tweet_id'

dfImage_predictions.tweet_id.nunique() == len(dfImage_predictions)
# are there any tweets where the algorithm couldn't make any prediction

dfImage_predictions.query('(p1_dog == False) & (p2_dog == False) & (p3_dog == False)')
dfTwitter_api.head()
dfTwitter_api.tail()
dfTwitter_api.info()
dfTwitter_api.describe()
# checking for duplicates in 'id'

dfTwitter_api.id.nunique() == len(dfTwitter_api.id)
# creating copy of original DataFrame before applying any changes

tArchive_clean = dfTwitter_archive.copy()
# 1. Harmonize names in `name` and replace non-sense naming with `none` for consistency

names = tArchive_clean['name'].str.contains('^[a-z]', regex = True)

tArchive_clean.loc[names, 'name'] = 'none'



# testing for new 'none' values

len(tArchive_clean.query('name == "none"'))
# 2. Drop all entries with missing URL that are **not** valid tweets

drop = tArchive_clean['retweeted_status_id'].notnull()

tArchive_clean.drop(tArchive_clean[drop].index, inplace=True)

tArchive_clean.reset_index(inplace=True, drop=True)



# testing if all values are gone

len(tArchive_clean[drop]) == 0
# 3. Drop all entries with missing URL that are **not** valid tweets

drop = (tArchive_clean['expanded_urls'].isnull() & tArchive_clean['in_reply_to_status_id'].notnull()) | (tArchive_clean['expanded_urls'].isnull() & tArchive_clean['retweeted_status_id'].notnull())



tArchive_clean.drop(tArchive_clean[drop].index, inplace=True)

tArchive_clean.reset_index(inplace=True, drop=True)



# testing if all values are gone

len(tArchive_clean[drop]) == 0
# 4. Drop all entries with `rating_numerator` > 15

drop = tArchive_clean['rating_numerator'] > 15



tArchive_clean.drop(tArchive_clean[drop].index, inplace=True)

tArchive_clean.reset_index(inplace=True, drop=True)



# testing if all values are gone

len(tArchive_clean[drop]) == 0
# 5. Set all entries with `rating_denominator` != 10 to `rating_denominator` = 10

m = tArchive_clean['rating_denominator'] != 10

tArchive_clean.loc[m, 'rating_denominator'] = 10



# testing if all values are gone

len(tArchive_clean[m]) == 0
# 6. Combine all four columns for dog stages into a single column stage 

stages = ['doggo', 'floofer', 'pupper', 'puppo']



# run loop to convert value from single column to centralized column 'stage'

for c in stages:

    tArchive_clean.loc[tArchive_clean[c] == c, 'stage'] = c

    

# filters for testing

filter1 = [(tArchive_clean[stages[0]] == stages[0]),

           (tArchive_clean[stages[1]] == stages[1]),

           (tArchive_clean[stages[2]] == stages[2]),

           (tArchive_clean[stages[3]] == stages[3])]



filter2 = [(tArchive_clean['stage'] == stages[0]),

            (tArchive_clean['stage'] == stages[1]),

            (tArchive_clean['stage'] == stages[2]),

            (tArchive_clean['stage'] == stages[3])]



# looping the test for all possible combinations



def test(df):

    options = 0

    for f in filter1:

        print('Test', options,':' , len(df[filter1[options]]) == len(df[filter2[options]]))

        options += 1



test(tArchive_clean)
# first test is 'False', meaning we have less/more 'doggo' values than before

len(tArchive_clean[filter1[0]]), len(tArchive_clean[filter2[0]])
# need to identify the difference in rows

errors = (tArchive_clean[stages[0]] == stages[0]) & (tArchive_clean['stage'] != stages[0])

len(tArchive_clean[errors])
# we found 14 values where we had two dog stages defined; for consistency, we will drop these

tArchive_clean.drop(tArchive_clean[errors].index, inplace=True)

tArchive_clean.reset_index(inplace=True, drop=True)
# running the test on the DataFrame again

test(tArchive_clean)
# drop redudant dog stage columns

tArchive_clean.drop(columns=stages, inplace=True)

tArchive_clean.reset_index(inplace=True, drop=True)
# 7. Convert the HTML-string from source into categories

iphone = tArchive_clean['source'].str.contains('Twitter for iPhone')

desktop = tArchive_clean['source'].str.contains('Twitter Web Client')

tweet_deck = tArchive_clean['source'].str.contains('TweetDeck')

vine = tArchive_clean['source'].str.contains('Vine - Make a Scene')



tArchive_clean.loc[iphone, 'source'] = 'iphone'

tArchive_clean.loc[desktop, 'source'] = 'desktop'

tArchive_clean.loc[tweet_deck, 'source'] = 'tweet_deck'

tArchive_clean.loc[vine, 'source'] = 'vine'



# testing for remaining unique values

tArchive_clean['source'].unique()
# 8. Convert tweet_id into string format; rename to ID

tArchive_clean['tweet_id'] = tArchive_clean['tweet_id'].astype(str)

tArchive_clean.rename(columns={'tweet_id':'ID'}, inplace=True)



# testing if datatype is now object

tArchive_clean['ID'].dtype == object
# 9. Convert timestamp into proper timestamp format

tArchive_clean['timestamp'] = pd.to_datetime(tArchive_clean['timestamp'])
# 10. Drop columns in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp



tArchive_clean.drop(columns=['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp'], inplace=True)



# testing if all columns are gone

tArchive_clean.info()
# creating copy of original DataFrame before applying any changes

iPredictions_clean = dfImage_predictions.copy()
# 1. Remove non-predictive rows; 324 in total

drop = (iPredictions_clean['p1_dog'] == False) & (iPredictions_clean['p2_dog'] == False) & (iPredictions_clean['p3_dog'] == False)



iPredictions_clean.drop(iPredictions_clean[drop].index, inplace=True)

iPredictions_clean.reset_index(inplace=True, drop=True)



len(iPredictions_clean[drop]) == 0
# 2. Rename tweet_id into a string and rename to ID for consistency

iPredictions_clean['tweet_id'] = iPredictions_clean['tweet_id'].astype(str)

iPredictions_clean.rename(columns={'tweet_id':'ID'}, inplace=True)



# testing if datatype is now object

iPredictions_clean['ID'].dtype == object
# 3. Extract most confident prediction and store it in column `breed` and `confidence_level` for the confidence

for i in range(len(iPredictions_clean)):

    if iPredictions_clean.loc[i, 'p1_dog'] == True:

        iPredictions_clean.loc[i, 'breed'] = iPredictions_clean.iloc[i]['p1']

        iPredictions_clean.loc[i, 'confidence_level'] = iPredictions_clean.iloc[i]['p1_conf']

    elif iPredictions_clean.loc[i, 'p2_dog'] == True:

        iPredictions_clean.loc[i, 'breed'] = iPredictions_clean.iloc[i]['p2']

        iPredictions_clean.loc[i, 'confidence_level'] = iPredictions_clean.iloc[i]['p2_conf']

    elif iPredictions_clean.loc[i, 'p3_dog'] == True:

        iPredictions_clean.loc[i, 'breed'] = iPredictions_clean.iloc[i]['p3']

        iPredictions_clean.loc[i, 'confidence_level'] = iPredictions_clean.iloc[i]['p3_conf']

        

# visual assessment/testing via tail; comparing if the conditions have been obeyed

iPredictions_clean.tail()
# creating copy of original DataFrame before applying any changes

tAPI_clean = dfTwitter_api.copy()
# 1. Rename `id` into a string and rename to `ID` for consistency

tAPI_clean['id'] = tAPI_clean['id'].astype(str)

tAPI_clean.rename(columns={'id':'ID'}, inplace=True)



# testing if datatype is now object

tAPI_clean['ID'].dtype == 'object'
# merging the 'breed' and 'cofidence_level' into the 'tArchive_clean' DataFrame

columns = ['ID', 'breed', 'confidence_level']

tArchive_master = pd.merge(tArchive_clean, iPredictions_clean[columns], on='ID', how='inner')
tArchive_master.head()
# merging the 'favorites' and 'retweets' into the 'tArchive_clean' DataFrame

columns = ['ID', 'favorites', 'retweets']

tArchive_master = pd.merge(tArchive_master, tAPI_clean[columns], on='ID', how='inner')
# last visual inspection before saving to master file

tArchive_master.head()
tArchive_master.tail()
# tArchive_master.to_csv('t_archive_master.csv', index = False)
# ls *.csv
# reading in the cleaned up file into our final DataFrame

df = pd.read_csv('/kaggle/input/twitter/t_archive_master.csv')

df.head()
source = df['source'].value_counts()

labels = df['source'].unique()



fig, ax = plt.subplots(figsize=(6, 5))

fig.subplots_adjust(0.3,0,1,1)



_, _ = ax.pie(source, startangle=90)



total = sum(source)

plt.legend(

    loc='upper left',

    labels=['%s, %1.1f%%' % (

        l, (float(s) / total) * 100) for l, s in zip(labels, source)],

    prop={'size': 11},

    bbox_to_anchor=(0.0, 1),

    bbox_transform=fig.transFigure

)



ax.set_title('Tweets per Source')

ax.axis('equal')

plt.show()
mu = df['rating_numerator'].mean()

sigma = df['rating_numerator'].std()

x = df['rating_numerator']



bins = 12



fig, ax = plt.subplots()



# histogram of the data

n, bins, patches = ax.hist(x, bins, density=1)



# additional 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')

ax.set_xlabel('Rating Numerator')

ax.set_ylabel('Relative Frequency')

ax.set_title('Distribution of Ratings')



fig.tight_layout()

plt.show()
# gathering some additional insights using some simple descriptive statistics

print('Mean:', df['rating_numerator'].mean()), 

print('Median:', df['rating_numerator'].median()), 

print('Mode:', statistics.mode(df['rating_numerator']))
# top five breeds in the DataFrame

b = df.breed.value_counts()

b[b > 50]
# what is the most frequent stage

df.stage.hist()

plt.ylabel('Frequency')

plt.xlabel('Stage')

plt.title('Frequency of Stages');
print('Total Favorites:', df.favorites.sum()), print('Total ReTweets:', df.retweets.sum())
x = df['timestamp']

y1 = df.rolling(window=50)['favorites'].mean()

y2 = df.rolling(window=50)['retweets'].mean()



plt.plot(x,y1, color='red', linewidth=1, label='Favorites')

plt.plot(x,y2, color='blue', linewidth=1, label='Retweets')

plt.xticks([])

plt.legend()

plt.xlabel('Time')

plt.ylabel('Amount of Retweets/Favorites')

plt.title('Engagement with Tweets over Time')

plt.show();