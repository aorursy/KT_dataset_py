import pandas as pd

IRAhandle_tweets_1 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_1.csv", encoding='latin1', nrows = 100)

IRAhandle_tweets_2 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_2.csv", encoding='latin1', nrows = 100)

IRAhandle_tweets_3 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_3.csv", encoding='latin1', nrows = 1000)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import gensim

from gensim import corpora, models, similarities

import logging

import tempfile

from nltk.corpus import stopwords

from string import punctuation

from collections import OrderedDict

import pyLDAvis.gensim

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, KFold

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer







import os
tweets1 = IRAhandle_tweets_1[:100]

tweets2 = IRAhandle_tweets_2[:100]

tweets3 = IRAhandle_tweets_3[:100]

tweets = pd.concat([tweets1, tweets2, tweets3])
tweets.head()
tweets.info()
import pandas_profiling

#tweets.profile_report()

#https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic
import plotly.express as px

fig = px.histogram(tweets, x="followers")

fig.show()
print("The average russian troll had " + str(tweets["followers"].mean()) + " followers.")

print("The median number of followers for russian trolls was: " + str(tweets["followers"].median()))

print("The most common number of followers for a russian troll to have was: " + str(tweets["followers"].mode()))
fig = px.histogram(tweets, x="following")

fig.show()
print("The average russian troll was following " + str(tweets["following"].mean()) + " accounts")

print("The median number of accounts followed by russian trolls was: " + str(tweets["following"].median()))
g = sns.heatmap(tweets[["followers","following"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
tweets['Time'] = pd.to_datetime(tweets['publish_date'])#, format='%-d/%-m/%y %H:%M')
fig = px.histogram(tweets, x="Time")

fig.show()
tweets["Time"].mean()
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)
tweets['content'] = tweets['content'].apply(remove_punctuation)

tweets.head(10)
sw = stopwords.words('english')

def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
tweets['content'] = tweets['content'].apply(stopwords)

tweets.head(10)
# create a count vectorizer object

count_vectorizer = CountVectorizer()

# fit the count vectorizer using the text data

count_vectorizer.fit(tweets['content'])

# collect the vocabulary items used in the vectorizer

dictionary = count_vectorizer.vocabulary_.items()



#this code is from a template from itrat on Kaggle
# this code continues to be from itrat



vocab = []

count = []

# iterate through each vocab and count append the value to designated lists

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index

vocab_after_stem = pd.Series(count, index=vocab)

# sort the dataframe

vocab_after_stem = vocab_after_stem.sort_values(ascending=False)

# plot of the top vocab

top_vacab = vocab_after_stem.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (15120, 15145))
tweets['tweet_length'] = tweets['content'].str.len()

print("The average tweet length is " + str(tweets['tweet_length'].mean()) + " characters long")

print("The median tweet length is " + str(tweets['tweet_length'].median()) + " characters long")
plt.boxplot('tweet_length',data = tweets)