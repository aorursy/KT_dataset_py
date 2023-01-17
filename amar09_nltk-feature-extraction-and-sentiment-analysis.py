import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# documents[10]
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]
# word_features[0:10]
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # will return either True or False

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# featuresets[0]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
data = pd.read_csv('../input/Sentiment.csv')
data.head()
# Keeping only the neccessary columns
data = data[['text','sentiment']]
data.head()
# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    # Filtering out the words with less than 4 characters     
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    
    # Here we are filtering out all the words that contains link|@|#|RT     
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    
    # filerting out all the stopwords 
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    
    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 
    tweets.append((words_without_stopwords, row.sentiment))
# It will extract all words from every tweets and will put it into seperate list
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    # extracted all the words only    
    return all

# Note that, we are not using this frequency of word occurance anywhere, So it will just return the unique word list. 
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

all_words_in_tweets = get_words_in_tweets(tweets)

w_features = get_word_features(all_words_in_tweets)

# w_features
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

wordcloud_draw(w_features)
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features[f'contains({word})'] = (word in document_words)
    return features
training_set = nltk.classify.apply_features(extract_features,tweets)
# training_set[0]
len(training_set), len(tweets)
len(training_set[0][0]), len(w_features)
classifier = nltk.NaiveBayesClassifier.train(training_set)
test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']
neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    
