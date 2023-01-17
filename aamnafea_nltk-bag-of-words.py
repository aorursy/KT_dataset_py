import numpy as np
import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

print(os.listdir("../input"))
stop = stopwords.words("english")
for char in [',','.',"'",'"','-', '(',')',':','?','/','>','<',"''", 'br', '\\','...']:
    stop.append(char)
train = pd.read_csv('../input/labeledTrainData.tsv', delimiter='\t')
test = pd.read_csv('../input/testData.tsv', delimiter='\t')
train.head()
test.head()
len(train)
len(nltk.word_tokenize(train['review'][0]))
review_sentiment = []
for id_ in train.index:
    review_sentiment.append([nltk.word_tokenize(train['review'][id_]), train['sentiment'][id_]])
review_sentiment = review_sentiment[:5000]
random.shuffle(review_sentiment)
bag_of_words = []
for (review,sentiment) in review_sentiment:
    for word in review:
        if word not in stop:
            bag_of_words.append(word)
len(bag_of_words)
word_FD = nltk.FreqDist(bag_of_words)
word_FD.most_common(10)
len(word_FD)
word_FD_cut = list(word_FD.keys())[:10000]
featuresets = []
for (review, sentiment) in review_sentiment:
    words = set(word for word in review \
                 if word not in stop)
    features= {}
    for w in word_FD_cut:
        features[w] =  w in words
    featuresets.append([features, sentiment])
len(featuresets)
train = featuresets[:4000]
valid = featuresets[4000:]
clf = nltk.NaiveBayesClassifier.train(train)
nltk.classify.accuracy(clf, valid)
clf.show_most_informative_features(10)
test.head(1)
test_reviews = []
for id_ in test.index:
    test_reviews.append(nltk.word_tokenize(test['review'][id_]))
len(test_reviews)
test_sets = []
for review in test_reviews:
    words = set(word for word in review \
                 if word not in stop)
    features= {}
    for w in word_FD_cut:
        features[w] =  w in words
    test_sets.append(features)
labels = clf.classify_many(test_sets)
len(labels)
submission = pd.DataFrame({"id": test['id'], "sentiment":labels})
submission.to_csv('submission.csv', index=False)






