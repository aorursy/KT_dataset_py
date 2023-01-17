# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
!pip install tweepy
import tweepy

# Twitter Developer keys here
# It is CENSORED
consumer_key = 'Rxk3CrnZEBNp9rnKVrLEtgb0n'
consumer_key_secret = 'fS246Gu7fd0A6ngeo1ZqCNKH3dH3Fg3vM32HVMAYr9cIi4BP00'
access_token = '1283244120127889408-k33Iv4YI5d53NrA7qTAKtGW1iWtKk4'
access_token_secret = '9vELsHFineQO3llVaq89oFI2gSf5xiyJwXrjMXjWL8r0i'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# The following codes were from the following website: https://medium.com/analytics-vidhya/fetch-tweets-using-their-ids-with-tweepy-twitter-api-and-python-ee7a22dcb845
# This method creates the training set
def createTrainingSet(corpusFile):
    import csv
    import time

    counter = 0
    corpus = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[0], "label": row[1], "topic": row[0]})

    trainingDataSet = []

    for tweet in corpus:
        try:
            tweetFetched = api.get_status(tweet["tweet_id"])
            print("Tweet fetched" + tweetFetched.text)
            tweet["text"] = tweetFetched.text
            dataset.append(tweet)
        except:
            continue


# Code starts here
# This is corpus dataset
corpusFile = "../input/april28-april29-twitter/april28_april29.csv"
# Call the method
resultFile = createTrainingSet(corpusFile)
