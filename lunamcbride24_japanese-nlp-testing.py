import sys #Used exclusively to get ja_core_news_lg, which does not save between sessions
!{sys.executable} -m spacy download ja_core_news_lg #Downloads ja_core_news_lg, which is used for spacy Japanese processing
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expressions
import html # HTML content, like &amp;

import spacy #For general NLP
from spacy.lang.ja.stop_words import STOP_WORDS #Get the Japanese stopwords

import ja_core_news_lg #Japanese language handling
nlp =  ja_core_news_lg.load() #Initializing Spacy for Japanese
import operator #For dictionary sorting

import matplotlib.pylab as plt #For plot testing
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tweetData = pd.read_csv("../input/ml-tweet/Twitter ML Data.csv") #Load the dataset into pandas
tweetData.head() #Take a peek at the dataset
tweetData = tweetData.rename(columns = {"Name" : "Tweeter", "2" : "OriginalTweet", "3" : "TweetLink", "4" : "TweetTime"}) #Changes the column names to something more helpful
tweetData.head() #Take a peek at the dataset
print(tweetData.count()) #Get the counts of values in the dataset

#Check each column if there are any null values
print(tweetData["Tweeter"].isnull().any())
print(tweetData["OriginalTweet"].isnull().any())
print(tweetData["TweetLink"].isnull().any())
print(tweetData["TweetTime"].isnull().any())
print(tweetData["5"].isnull().any())
print(tweetData["6"].isnull().any())
tweetData = tweetData.drop(columns = ["5", "6"]) #Drop the null columns
tweetData.head() #Take a peek at the dataset
tweetData["OriginalTweet"] = tweetData["OriginalTweet"].apply(lambda x: np.nan if not x else x) #Change blank tweets to null as well
tweetData.dropna(subset = ["OriginalTweet"], inplace = True) #Drop the nulls based on the tweet data
tweetData.reset_index(drop=True, inplace=True) #Reset the index for later looping
tweetData.head() #Take a peek at the dataset
#Check each column if there are any null values
print(tweetData["Tweeter"].isnull().any())
print(tweetData["OriginalTweet"].isnull().any())
print(tweetData["TweetLink"].isnull().any())
print(tweetData["TweetTime"].isnull().any())
#Modified from my previous nlp project: https://www.kaggle.com/lunamcbride24/coronavirus-tweet-processing

punctuations = """!()（）「」、-!！[]{};:+'"\,<>./?@#$%^&*_~Â。…・，【】？""" #List of punctuations to remove, including a weird A that will not process out any other way

stopwords = spacy.lang.ja.stop_words.STOP_WORDS

#CleanTweets: parces the tweets and removes punctuation, stop words, digits, and links.
#Input: the list of tweets that need parsing
#Output: the parsed tweets
def cleanTweets(tweetParse):
    for i in range(0,len(tweetParse)):
        tweet = tweetParse[i] #Putting the tweet into a variable so that it is not calling tweetParse[i] over and over
        tweet = html.unescape(tweet) #Removes leftover HTML elements, such as &amp;
        tweet = re.sub(r"RT", ' ', tweet)
        tweet = re.sub(r"\n", ' ', tweet)
        tweet = re.sub(r"@\w+", ' ', tweet) #Completely removes @'s, as other peoples' usernames mean nothing
        tweet = re.sub(r'https\S+', ' ', tweet) #Removes links, as links provide no data in tweet analysis in themselves
        tweet = re.sub(r"\d+", ' ', tweet) #Removes numbers, as well as cases like the "th" in "14th"
        tweet = ''.join([punc for punc in tweet if not punc in punctuations]) #Removes the punctuation defined above
        tweet = tweet.lower() #Turning the tweets lowercase real quick for later use
    
        tweetWord = nlp.tokenizer(tweet) #Splits the tweet into individual words
        tweetParse[i] = ''.join([word.orth_ + " " for word in tweetWord if word.is_stop == False]) #Checks if the words are stop words
       
        
    return tweetParse #Returns the parsed tweets

tweets = tweetData["OriginalTweet"].copy() #Gets a copy of the tweets to send to the function call
tweetData["CleanTweet"] = cleanTweets(tweets) #Adds a CleanTweet column and fills it with processed tweets
print(tweetData["OriginalTweet"][3], "\n \n", tweetData["CleanTweet"][3]) #Prints an example sentence
tweetData.head() #Takes a peek at the dataframe
print(list(stopwords)) #Print the stopwords for reference
tweets = tweetData["CleanTweet"].copy() #Copy the clean tweets for processing
count = dict() #Creates a dictionary 
for i in range(0,len(tweets)):
    words = tweets[i].split()
    for word in words:
        if word in count:
            count[word] += 1
        else:
            count[word] = 1

sortCount = {word : summ for word, summ in sorted(count.items(), key=operator.itemgetter(1),reverse=True)}
print(dict(list(sortCount.items())[0: 20]))
top = dict(list(sortCount.items())[0: 20])
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
plt.bar(top.keys(), top.values(), color = "g")