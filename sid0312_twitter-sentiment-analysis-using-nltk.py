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
import matplotlib.pyplot as plt

%matplotlib inline

import nltk

import re
from nltk.tokenize import word_tokenize
data = pd.read_csv('/kaggle/input/twitter-sentiment-analysis/train_E6oV3lV.csv')
data.head()
y_train = list(data['label'])
tweets = list(data['tweet'])
good=data['label'].value_counts()[0]

bad = data['label'].value_counts()[1]
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

labels = ['good', 'bad']

quantity = [good,bad]

ax.bar(labels,quantity)

plt.xlabel('Quality of tweet')

plt.ylabel('Number of tweets')

plt.show()
data[data['label'] == 1].head(10)


def remove_special_characters(text, remove_digits=True):

    pattern=r'[^a-zA-z\s]'

    text=re.sub(pattern,'',text)

    return text



data['tweet']=data['tweet'].apply(remove_special_characters)

k=data['tweet'][0]
#tokenize

t=[]

for tweet in list(data['tweet']):

    tokened=word_tokenize(tweet)

    t.append(tokened)

#removing the stopwords

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

k=[]

for tweet in t:

    filtered_sentence = [w for w in tweet if not w in stop_words]

    k.append(filtered_sentence)

#stemming

from nltk.stem import PorterStemmer

ps = PorterStemmer()

f=[]

for tweet in k:

    x=[ps.stem(word) for word in tweet]

    f.append(x)

        
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

h=[]

for tweet in f:

    x=[lemmatizer.lemmatize(word) for word in tweet]

    h.append(x)
data['normalized_tweet']=h
def check(a):

    if a==0:

        return 'positive tweet'

    else:

        return 'negative tweet'
m=[]

for tweet in h:

    r= [word for word in tweet if word !='user']

    m.append(r)
all_words = []



for tweet in m:

    for word in tweet:

        all_words.append(word.lower())



all_words = dict(nltk.FreqDist(all_words))
a = sorted(all_words.items(), key=lambda x: x[1],reverse=True) 
l=[]

for x,y in a:

    l.append(x)
word_features=l[:3000]
documents=[]

i=0

for tweet in m:

    x=(tweet,check(data['label'][i]))

    i=i+1

    documents.append(x)
documents[0]
def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features
featuresets = [(find_features(tweet), sentiment) for (tweet, sentiment) in documents]
featuresets[0]
len(featuresets)
training_set = featuresets[:25000]

testing_set = featuresets[25000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100,"%",sep='')

classifier.show_most_informative_features(30)
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())

from sklearn.svm import SVC, LinearSVC, NuSVC

MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100,"%",sep="")



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100,"%",sep="")



SVC_classifier = SklearnClassifier(SVC())

SVC_classifier.train(training_set)

print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100,"%",sep="")



LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100,"%",sep="")
