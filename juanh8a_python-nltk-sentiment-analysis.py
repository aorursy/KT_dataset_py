# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Sentiment.csv')

data = data[['text','sentiment']]

data.text = data.text.str.replace('@','')

data.text = data.text.str.replace('#','')

train, test = train_test_split(data,test_size = 0.1)

train = train[train.sentiment != "Neutral"]

train = train[0:1500]

train_pos = []

train_neg = []



for index, row in train.iterrows():

    if row['sentiment'] == "Positive":

        train_pos.append(row)

    elif row['sentiment'] == "Negative":

        train_neg.append(row)



tweets = []

for index, row in train.iterrows():

    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]

    tweets.append((words_filtered,row.sentiment))



test_pos = []

test_neg = []



for index, row in test.iterrows():

    if row['sentiment'] == "Positive":

        test_pos.append(row)

    elif row['sentiment'] == "Negative":

        test_neg.append(row)
def get_words_in_tweets(tweets):

    all = []

    for (words, sentiment) in tweets:

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

        features['containts(%s)' % word] = (word in document_words)

    return features
training_set = nltk.classify.apply_features(extract_features,tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)
neg_cnt = 0

pos_cnt = 0

for obj in test_neg: 

    res =  classifier.classify(extract_features(obj.text.split()))

    if(res == 'Negative'): 

        neg_cnt = neg_cnt + 1

for obj in test_pos: 

    res =  classifier.classify(extract_features(obj.text.split()))

    if(res == 'Positive'): 

        pos_cnt = pos_cnt + 1

        

print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        

print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    