# Following the instructions from https://www.toptal.com/python/twitter-data-mining-using-python

# I created Twitter Developer Account and Application, and acquired consumer API key and access

# token, which is used by tweepy for authenticating with Twitter before data access is granted



# Run 'pip install tweepy' in Console before running next line

import tweepy

import json



consumer_key = "FsBLgaWgsldl7AkCMLJ6U8r9y"

consumer_secret = "fCoRUEWWSucBYa3TpqGn1a2PFrFbntswhex1nFrnGwY7vRAGLH"

access_token = "4740522320-Mxz1rSf1TjWw7GeV3ZHCDeQBRg1G4Vd01qloH3y"

access_token_secret = "XvvA1gRK2dJxjquN8EV2cUWaM7S4wAJfzRpBrBpaV5LkH"



# Creating authentication information to initialize API client

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth) 



# To verify the API client can access Twitter data

# Try to fetch the 3 latest tweets from my own account (@tfang89) 

results = api.user_timeline(id="tfang89", count=3)

for tweet in results:

    # format the json data from each tweet for pretty print

    tweet_json = json.dumps(tweet._json, sort_keys=True, indent=2)

    print(tweet_json)
# Now that I verified that API client is working correctly,

# I can move on to query for data related to the subject

# Obviously Twitter API returns a massive list of variables for each tweet,

# but for the purpose of this analysis, only text field is what matters



import random

import re



# Search using full term 'deltaairlines', which returned 307 results only

print('***************************************')

print('Try searching for deltaairlines')



delta_result, delta_count = [], 0

for tweet in tweepy.Cursor(api.search, 

                           q="deltaairlines", 

                           count=100, 

                           result_type="recent", 

                           lang="en").items():

    delta_result.append(tweet.text)

    delta_count += 1



print('Search using deltaairlines, collected %s tweets.' % len(delta_result))



print('Samples:')

for sample in range(10):

    print(random.choice(delta_result))



# Search using Delta Airlines' twitter handle '@delta', which returned a lot more results

# Also perform the operation to remove links

print('***************************************')

print('Try searching for @delta')



delta_result, delta_count = [], 0

for tweet in tweepy.Cursor(api.search, 

                           q="%40delta", 

                           count=100, 

                           result_type="recent", 

                           lang="en", 

                           tweet_mode='extended').items():

    try:

        tweet_text = tweet.retweeted_status.full_text

    except AttributeError:  # Not a Retweet

        tweet_text = tweet.full_text

    tweet_text = re.sub(r"http\S+", "", tweet_text)

    delta_result.append(tweet_text)

    delta_count += 1

    if delta_count >= 500:

        break



print('Search using @delta, collected at least %s tweets.' % len(delta_result))



print('Samples:')

for sample in range(10):

    print(random.choice(delta_result))
# The samples above finally looks good

# Now repeat the process to collect 10K tweets for each airline

# To avoid hitting Twitter's API data limit (which triggers HTTP error 429)

# I'm also saving the result into CSV file



import pandas as pd



def collectTweets(queryStr, limit):

    result, count = [], 0

    for tweet in tweepy.Cursor(api.search, 

                               q=queryStr, 

                               count=100, 

                               result_type="recent", 

                               lang="en", 

                               tweet_mode='extended').items():

        try:

            tweet_text = tweet.retweeted_status.full_text

        except AttributeError:  # Not a Retweet

            tweet_text = tweet.full_text

        tweet_text = re.sub(r"http\S+", "", tweet_text)

        result.append(tweet_text)

        count += 1

        if count >= limit:

            break

    return result
# I've collected data for Delta already saved in delta.csv

# Do not call this function again which will cause other queries to be throttled

# print('Starting to collect tweets for Delta Airlines')

# delta_result = collectTweets('%40Delta', 10000)



print('Successfully collected %s tweets for Delta Airlines' % len(delta_result))

print('Writing to delta.csv')

df = pd.DataFrame(delta_result, columns=["tweets"])

df.to_csv('delta.csv', index=False)
print('Starting to collect tweets for United Airlines')

united_result = collectTweets('%40united', 10000)



print('Successfully collected %s tweets for United Airlines' % len(delta_result))

print('Writing to united.csv')

df = pd.DataFrame(united_result, columns=["tweets"])

df.to_csv('united.csv', index=False)
print('Starting to collect tweets for American Airlines')

american_result = collectTweets('%40AmericanAir', 10000)



print('Successfully collected %s tweets for American Airlines' % len(delta_result))

print('Writing to american.csv')

df = pd.DataFrame(american_result, columns=["tweets"])

df.to_csv('american.csv', index=False)
# Frist, making sure all three data files have been written before running this code



import os

import errno

import numpy as np

import pandas as pd



filenames = []



# When collecting tweets

# current_path = '/kaggle/working/'



# When reusing tweets collected earlier

current_path = '/kaggle/input/big-three-airline-tweets/'



for dirname, _, files in os.walk(current_path):

    for file in files:

        filenames.append(file)



if 'delta.csv' not in filenames:

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'delta.csv')

if 'united.csv' not in filenames:

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'united.csv')

if 'american.csv' not in filenames:

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'american.csv')
# Let's use Delta as example for analysis, and use TextBlob to 

# analyze the sentiment for each tweet



from textblob import TextBlob



def analyzeSentiment(tweet):

    senti = TextBlob(tweet).sentiment

    return senti.polarity



pd.set_option('display.max_colwidth', -1)



delta_data = pd.read_csv(current_path + 'delta.csv')



# create new polarity column by applying the TextBlob sentiment analysis

delta_data['polarity'] = delta_data['tweets'].apply(analyzeSentiment)
# sample some highly positive tweets

delta_positive = delta_data[delta_data.polarity > 0.75].sample(5)

delta_positive.head()
# sample some highly negative tweets

delta_negative = delta_data[delta_data.polarity < -0.75].sample(5)

delta_negative.head()
# result looks good, now let's count the tweets by categories for visualization



import operator



def countTweetsBySentiment(airline_df):

    highly_positive = airline_df[airline_df.polarity >= 0.75]

    positive = airline_df[operator.and_(airline_df.polarity >= 0.25, airline_df.polarity < 0.75)]

    neutral = airline_df[operator.and_(airline_df.polarity > -0.25, airline_df.polarity < 0.25)]

    negative = airline_df[operator.and_(airline_df.polarity > -0.75, airline_df.polarity <= -0.25)]

    highly_negative = airline_df[airline_df.polarity <= -0.75]

    

    return [highly_positive.shape[0], 

            positive.shape[0], 

            neutral.shape[0], 

            negative.shape[0], 

            highly_negative.shape[0]]





delta_result = countTweetsBySentiment(delta_data)

print('Delta tweets count by sentiment:')

print(delta_result)



# perform the same analysis on United and American Airlines



united_data = pd.read_csv(current_path + 'united.csv')

united_data['polarity'] = united_data['tweets'].apply(analyzeSentiment)

united_result = countTweetsBySentiment(united_data)

print('United tweets count by sentiment:')

print(united_result)



american_data = pd.read_csv(current_path + 'american.csv')

american_data['polarity'] = american_data['tweets'].apply(analyzeSentiment)

american_result = countTweetsBySentiment(american_data)

print('American tweets count by sentiment:')

print(american_result)

# Start visualization with pie chart for each airline



import matplotlib.pyplot as plt



labels = ['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Highly Negative']

explode = (0.5, 0.2, 0, 0.2, 0.5)  # explode 'Highly Positive' and 'Highly Negative'



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,15))

ax1.pie(delta_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title('Delta Airlines')



ax2.pie(united_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title('United Airlines')



ax3.pie(american_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax3.set_title('American Airlines')



plt.show()
# Try second visualization which is stacked bar plot



stack_df = pd.DataFrame()

stack_df = pd.DataFrame({'Delta': delta_result,

                        'United': united_result, 

                        'American': american_result}, 

                        index=labels)



stack_df.loc[:,['Delta','United', 'American']].plot.bar(stacked=True, figsize=(10,7))
# This plot is not very readable, try improve it by ploting using another dimention



labels = ['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Highly Negative']



stack_df = pd.DataFrame()

stack_df = pd.DataFrame({'Highly Positive': [delta_result[0], united_result[0], american_result[0]],

                        'Positive': [delta_result[1], united_result[1], american_result[1]],

                        'Neutral': [delta_result[2], united_result[2], american_result[2]],

                        'Negative': [delta_result[3], united_result[3], american_result[3]],

                        'Highly Negative': [delta_result[4], united_result[4], american_result[4]],}, 

                        index=['Delta', 'United', 'American'])



stack_df.loc[:,['Highly Positive', 'Positive', 'Neutral', 'Negative', 'Highly Negative']].plot.bar(stacked=True, figsize=(10,7))
# The first step is to filter the negative tweets from each airlines, 

# and split into train and test set



def writeToTrainAndTest(prefix, airline_df):

    count = airline_df.shape[0]

    train_num = min(0.2 * count, 100)

    test_num = count - train_num

    train = airline_df.head(train_num)

    print('Writing {0} samples into training set {1}'.format(train_num, prefix + '_train.csv'))

    train.to_csv(prefix + '_train.csv', index=False)

    test = airline_df.tail(test_num)

    print('Writing {0} samples into test set {1}'.format(test_num, prefix + '_test.csv'))

    test.to_csv(prefix + '_test.csv', index=False)



delta_negative = delta_data[delta_data.polarity <= -0.25].loc[:,['tweets']]

united_negative = united_data[united_data.polarity <= -0.25].loc[:,['tweets']]

american_negative = american_data[american_data.polarity <= -0.25].loc[:,['tweets']]



writeToTrainAndTest('delta', delta_negative)

writeToTrainAndTest('united', united_negative)

writeToTrainAndTest('american', american_negative)
# To fit into my monthly budge of 299 queries for this month

# I need to sample the test data set



delta_df = pd.read_csv('/kaggle/working/delta_test.csv')

delta_sampled = delta_df.sample(100)

delta_sampled.to_csv('delta_test_sample.csv', index=False)



united_df = pd.read_csv('/kaggle/working/united_test.csv')

united_sampled = united_df.sample(100)

united_sampled.to_csv('united_test_sample.csv', index=False)



american_df = pd.read_csv('/kaggle/working/american_test.csv')

american_sampled = american_df.sample(99)

american_sampled.to_csv('american_test_sample.csv', index=False)
# Load the processed result

delta_processed = pd.read_csv(current_path + 'delta_processed_batch.csv')

united_processed = pd.read_csv(current_path + 'united_processed_batch.csv')

american_processed = pd.read_csv(current_path + 'american_processed_batch.csv')



# Examine the data to make sure it looks good

delta_processed.head()
# Visualize the result using pie chart like I did earlier



import matplotlib.pyplot as plt



def countTweetsByTag(airline_df):

    delay = airline_df[airline_df.Classification == 'Delay']

    in_flight = airline_df[airline_df.Classification == 'In-flight Experience']

    luggage = airline_df[airline_df.Classification == 'Luggage']

    customer_service = airline_df[airline_df.Classification == 'Customer Service']

    price_fee = airline_df[airline_df.Classification == 'Price Fee']

    unknown = airline_df[airline_df.Classification == 'Unknown']

    

    return [delay.shape[0], 

            in_flight.shape[0], 

            luggage.shape[0], 

            customer_service.shape[0], 

            price_fee.shape[0],

            unknown.shape[0]]



delta_result = countTweetsByTag(delta_processed)

print('Delta tweets count by tags:')

print(delta_result)



united_result = countTweetsByTag(united_processed)

print('United tweets count by tags:')

print(united_result)



american_result = countTweetsByTag(american_processed)

print('American tweets count by tags:')

print(american_result)



labels = ['Delay', 'In-flight Experience', 'Luggage', 'Customer Service', 'Price Fee', 'Unknown']

explode = (0.1, 0.2, 0, 0.2, 0.1, 0)  # explode 'In-flight Experience' and 'Customer Service'



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,15))

ax1.pie(delta_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title('Delta Airlines')



ax2.pie(united_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title('United Airlines')



ax3.pie(american_result, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax3.set_title('American Airlines')



plt.show()