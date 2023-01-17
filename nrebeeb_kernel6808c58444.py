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
Tweet= pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")

Tweet.head()
Tweet.count()
print(Tweet['airline_sentiment'].value_counts())
del Tweet['airline_sentiment_gold']

del Tweet['negativereason_gold']

del Tweet['retweet_count']

del Tweet['tweet_coord']

#we dont need them
import matplotlib.pyplot as plt

Mood_count=Tweet['airline_sentiment'].value_counts()

Index = [1,2,3]

plt.bar(Index,Mood_count)

plt.xticks(Index,['negative','neutral','positive'],rotation=45)

plt.ylabel('Mood Count')

plt.xlabel('Mood')

plt.title('Count of Moods')

#study this
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

import nltk

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *







nltk.download('wordnet')

stemmer = SnowballStemmer("english")



def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))



# Tokenize and lemmatize

def preprocess(text):

    result=[]

    for token in gensim.utils.simple_preprocess(text) :

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

            

    return result

doc_sample = "I don't know how to use my new laptop, any advice?"

print(preprocess(doc_sample))
preporcessd_words = []

all_data_together = []

def preprocess_data (tweet_text):

    preporcessd_words = []

    for example in tweet_text:

        example_words = []

        for word in example.split(' '):

            preporcessd_word = preprocess(word)

            if preporcessd_word:

                for word in preporcessd_word:

                    example_words.append(word)

                    #all_data_together.append(word)

        #print example_words

        preporcessd_words.append(example_words)

    return preporcessd_words
#change value from "positive, nigative" to 0,1

Tweet['sentiment']=Tweet['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
import sklearn

from sklearn.model_selection import train_test_split

train,test = train_test_split(Tweet,test_size=0.2,random_state=42)
train_tweets = preprocess_data(train['text'])

test_tweets = preprocess_data(test['text'])

#print(test_tweets)

dictionary = gensim.corpora.Dictionary(preporcessd_words)

train_tweets_test = np.array(train_tweets)

#print (type(np.asarray(bow_corpus)))

bow_corpus = [train_tweets_test.doc2bow(doc) for doc in train_tweets]

#print (type(bow_corpus))

from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(bow_corpus)

#test_features=v.transform(test_tweets)

#print(test_tweets)
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



Classifiers = [SVC(kernel="rbf", C=0.025, probability=True), DecisionTreeClassifier()]

Accuracy=[]

Model=[]

for classifier in Classifiers:



    fit = classifier.fit(train_features,train['sentiment'])

    pred = fit.predict(test_features)

    accuracy = accuracy_score(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
