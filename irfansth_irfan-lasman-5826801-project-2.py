import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#print(os.listdir("../input/"))

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
from subprocess import check_output 
print(check_output(["ls", "../input/"]).decode("utf8"))
path = "../input/Musical_Instruments_5.json"
data = pd.read_json(path, lines=2)
data = data[['asin','reviewText','overall','summary']]
print(data)
# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral summary
train = train[train.summary != "Neutral"]
print(train)
train_pos = train[ train['summary'] == 'Positive']
train_pos = train_pos['reviewText']
train_neg = train[ train['summary'] == 'Negative']
train_neg = train_neg['reviewText']

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

#print("Positive words")
#wordcloud_draw(train_pos,'white')
#print("Negative words")
#wordcloud_draw(train_neg,)   
tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.reviewText.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.summary))

test_pos = test[ test['summary'] == 'Positive']
test_pos = test_pos['reviewText']
test_neg = test[ test['summary'] == 'Negative']
test_neg = test_neg['reviewText']
# Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, summary) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

wordcloud_draw(w_features)
# Training the Naive Bayes classifier
#training_set = nltk.classify.apply_features(extract_features,tweets)
#classifier = nltk.NaiveBayesClassifier.train(training_set)
#neg_cnt = 0
#pos_cnt = 0
#for obj in test_neg: 
#    res =  classifier.classify(extract_features(obj.split()))
#    if(res == 'Negative'): 
#        neg_cnt = neg_cnt + 1
#for obj in test_pos: 
#    res =  classifier.classify(extract_features(obj.split()))
#    if(res == 'Positive'): 
#        pos_cnt = pos_cnt + 1
#        
#print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
#print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    