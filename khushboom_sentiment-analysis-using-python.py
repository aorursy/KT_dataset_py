# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")

df.head()
list(df.columns.values)
senti_counts = df.airline_sentiment.value_counts()

print(senti_counts)

#we can see that the data is imbalanced
tweets_count = df.tweet_id.count()

print(tweets_count)
import nltk

import re

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
# removed stop words and performed lemmatization

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()
def normalize(tweet):

    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 

    tokens = nltk.word_tokenize(only_letters)[2:]

    lower_case = [l.lower() for l in tokens]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]

    return lemmas
# an example of normalize function

normalize("Would like to provide a feedback for the worst travel in my life")
pd.set_option('display.max_colwidth', -1) 

df['normalized_tweet'] = df.text.apply(normalize)

df[['text','normalized_tweet']].head()
from nltk import ngrams

def ngrams(input_list):

    onegrams = input_list

    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]

    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]

    return bigrams+trigrams

df['grams'] = df.normalized_tweet.apply(ngrams)

df[['grams']].head()
NegativeReason_Count=dict(df['negativereason'].value_counts(sort=False))
def NR_Count(Airline):

    if Airline=='All':

        d=df

    else:

        d=df[df['airline']==Airline]

    count=dict(df['negativereason'].value_counts())

    Unique_reason=list(df['negativereason'].unique())

    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']

    Reason_frame=pd.DataFrame({'R':Unique_reason})

    Reason_frame['count']=Reason_frame['R'].apply(lambda x: count[x])

    return Reason_frame
def plot_reason(Airline):

    d=NR_Count(Airline)

    count=d['count']

    Index = range(1,(len(d)+1))

    plt.bar(Index,count)

    plt.xticks(Index,d['R'],rotation=90)

    plt.ylabel('Count')

    plt.xlabel('Reason')

    plt.title('Count of Reasons for '+Airline)
plot_reason('All')

def avg_words(tweet):

    words=tweet.split()

    return (sum(len(word) for word in words)/len(words))

df['avg_words']=df['text'].apply(lambda x: avg_words(x))

df[['text','avg_words']].head()
tweet_count = df['airline'].sort_index().value_counts()

tweet_count.plot(kind='bar', rot=0)

plt.title('No of tweets per Airline')

plt.show()
from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer



count_vectorizer = CountVectorizer(ngram_range=(1,2))

vectorized_tweets = count_vectorizer.fit_transform(df['text'])

input_data = hstack((np.array(range(0,vectorized_tweets.shape[0]))[:,None], vectorized_tweets))
def senti_outcome(sentiment):

    return {

        'negative': 1,

        'neutral': 0,

        'positive' : 0

    }[sentiment]

outcome = df['airline_sentiment'].apply(senti_outcome)

print(outcome)
from sklearn.model_selection import train_test_split

train_data, test_data,train_outcome, test_outcome = train_test_split(input_data, outcome, test_size=0.3, random_state=0)

train_data = train_data[:,1:]

test_data = test_data[:,1:]
print(test_data)

print(test_outcome)

np.shape(test_data)

np.shape(test_outcome)

test_outcome.unique()

test_outcome.value_counts()
from sklearn.linear_model import LogisticRegression

logist_reg = LogisticRegression(C=0.6,solver='liblinear',max_iter=2000)

logist_reg_output =logist_reg.fit(train_data, train_outcome)

logist_reg_output.score(test_data, test_outcome)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score
y_pred=logist_reg.predict(test_data)

lr1_confusion=confusion_matrix(test_outcome, y_pred)

print('Logistic1 confusion matrix')

print(lr1_confusion)

lr1_recall=recall_score(test_outcome, y_pred)

print('recall for logistic regression 1')

print(lr1_recall)
from sklearn import svm

from sklearn.multiclass import OneVsRestClassifier



OVR = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100, probability=True, class_weight='balanced', kernel='linear'))

OVR_output = OVR.fit(train_data, train_outcome)

#evaluating the result

OVR.score(test_data, test_outcome)
y_pred= OVR.predict(test_data)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score

svm_confusion=confusion_matrix(test_outcome, y_pred)

print('SVM confusion matrix')

print(svm_confusion)

svm_recall=recall_score(test_outcome, y_pred)

print('recall for svm')

print(svm_recall)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

rf= RandomForestClassifier(n_estimators=500)

rf_output = rf.fit(train_data, train_outcome)

rf_output.score(test_data, test_outcome)
y_pred=rf.predict(test_data)

rf_confusion=confusion_matrix(test_outcome, y_pred)

print('Random Forest confusion matrix')

print(rf_confusion)

rf_recall=recall_score(test_outcome, y_pred)

print('recall for random forest')

print(rf_recall)
AB= AdaBoostClassifier()

AB_output = AB.fit(train_data, train_outcome)

AB_output.score(test_data, test_outcome)
y_pred=AB.predict(test_data)

ab_confusion=confusion_matrix(test_outcome, y_pred)

print('AdaBoost confusion matrix')

print(ab_confusion)

ab_recall=recall_score(test_outcome, y_pred)

print('recall for AdaBoost')

print(ab_recall)
# Since we want to classify negative tweets accurately, we are more interested in Recall score

# Looking at the recall score, we can say that random forest is giving highest score. 