# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Reference code: https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

from nltk.tokenize import TweetTokenizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



cat = ['sample_submission.csv','test.csv','train.csv']

data_cat = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(filename)

        path = os.path.join(dirname, filename)

        data_cat.append(pd.read_csv(path))



# Any results you write to the current directory are saved as output.
sub = data_cat[0]

test = data_cat[1]

train = data_cat[2]
# Vectorize dataset

# Credit: https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-4-count-vectorizer-b3f4944e51b5

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

  



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from time import time

lemmatizer = WordNetLemmatizer() 

stop = stopwords.words('english')
import re

from nltk.tokenize import word_tokenize

from string import punctuation 

from nltk.corpus import stopwords 



class PreProcessTweets:

    def __init__(self):

        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

        

    def processTweets(self, list_of_tweets):

        processedTweets=[]

        for index,tweet in list_of_tweets.iterrows():

            tokenized_tweets=self._processTweet(tweet["text"])

            try:

                t = int(tweet["target"])

            except:

                t = 0

            processedTweets.append((tokenized_tweets,t))

        return processedTweets

    

    def _processTweet(self, tweet):

        tweet = tweet.lower() # convert text to lower-case

        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs

        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames

        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag

        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)

        return [word for word in tweet if word not in self._stopwords]
tweetProcessor = PreProcessTweets()

preprocessedTrainingSet = tweetProcessor.processTweets(train)

preprocessedTestSet = tweetProcessor.processTweets(test)


import nltk 



def buildVocabulary(preprocessedTrainingData):

    all_words = []

    

    for (words, sentiment) in preprocessedTrainingData:

        all_words.extend(words)



    wordlist = nltk.FreqDist(all_words)

    word_features = wordlist.keys()

    

    return word_features





def extract_features(tweet):

    tweet_words = set(tweet)

    features = {}

    for word in word_features:

        features['contains(%s)' % word] = (word in tweet_words)

    return features 





word_features = buildVocabulary(preprocessedTrainingSet)

trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedTrainingSet)
import nltk 



def buildVocabulary(preprocessedTrainingData):

    all_words = []

    

    for (words, sentiment) in preprocessedTrainingData:

        all_words.extend(words)



    wordlist = nltk.FreqDist(all_words)

    word_features = wordlist.keys()

    

    return word_features

def extract_features(tweet):

    tweet_words=set(tweet)

    features={}

    for word in word_features:

        features['contains(%s)' % word]=(word in tweet_words)

    return features 

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]

final = pd.DataFrame({'id':test['id'],'target':NBResultLabels})
final[final['target']==1].count()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'disaster.csv'



final.to_csv(filename,index=False)



print('Saved file: ' + filename)
# x  = ['Forest', 'fire', 'nearing', 'La', 'Ronge', 'Sask', 'Canada']
# x = list(all_texts_lemmied[2])
# [lemmy.lemmatize(i) for i in x]
# def get_wordnet_pos(word):

#     """Map POS tag to first character lemmatize() accepts"""

#     tag = nltk.pos_tag([word])[0][1][0].upper()

#     tag_dict = {"J": wordnet.ADJ,

#                 "N": wordnet.NOUN,

#                 "V": wordnet.VERB,

#                 "R": wordnet.ADV}



#     return tag_dict.get(tag, wordnet.NOUN)
# print(lemmy.lemmatize("cats"))

# print(lemmy.lemmatize("cacti"))

# print(lemmy.lemmatize("geese"))

# print(lemmy.lemmatize("rocks"))

# print(lemmy.lemmatize("python"))

# print(lemmy.lemmatize("better", pos="a"))

# print(lemmy.lemmatize("best", pos="a"))

# print(lemmy.lemmatize("run"))

# print(lemmy.lemmatize("run",'v'))