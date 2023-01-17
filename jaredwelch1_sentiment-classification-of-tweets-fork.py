# Before we begin, we supress deprecation warnings resulting from nltk on Kaggle

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd

tweets = pd.read_csv("../input/Tweets.csv")

list(tweets.columns.values)
tweets.head()
sentiment_counts = tweets.airline_sentiment.value_counts()

number_of_tweets = tweets.tweet_id.count()

print(sentiment_counts)
import re, nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()



def normalizer(tweet):

    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 

    tokens = nltk.word_tokenize(only_letters)[2:]

    lower_case = [l.lower() for l in tokens]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]

    return lemmas
normalizer("I recently wrote some texts.")
pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells

tweets['normalized_tweet'] = tweets.text.apply(normalizer)

tweets[['text','normalized_tweet']].head()
from nltk import ngrams

def ngrams(input_list):

    #onegrams = input_list

    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]

    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]

    return bigrams+trigrams

tweets['grams'] = tweets.normalized_tweet.apply(ngrams)

tweets[['grams']].head()
import collections

def count_words(input):

    cnt = collections.Counter()

    for row in input:

        for word in row:

            cnt[word] += 1

    return cnt
tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)
tweets[(tweets.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20)
import numpy as np

from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(tweets.text)

indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))
def sentiment2target(sentiment):

    return {

        'negative': 0,

        'neutral': 1,

        'positive' : 2

    }[sentiment]

targets = tweets.airline_sentiment.apply(sentiment2target)
from sklearn.model_selection import train_test_split

data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)

data_train_index = data_train[:,0]

data_train = data_train[:,1:]

data_test_index = data_test[:,0]

data_test = data_test[:,1:]
from sklearn import svm

from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))

clf_output = clf.fit(data_train, targets_train)
clf.score(data_test, targets_test)
sentences = count_vectorizer.transform([

    "What a great airline, the trip was a pleasure!",

    "My issue was quickly resolved after calling customer support. Thanks!",

    "What the hell! My flight was cancelled again. This sucks!",

    "Service was awful. I'll never fly with you again.",

    "You fuckers lost my luggage. Never again!",

    "I have mixed feelings about airlines. I don't know what I think.",

    ""

])

clf.predict_proba(sentences)
predictions_on_test_data = clf.predict_proba(data_test)

index = np.transpose(np.array([range(0,len(predictions_on_test_data))]))

indexed_predictions = np.concatenate((predictions_on_test_data, index), axis=1).tolist()
def margin(p):

    top2 = p.argsort()[::-1]

    return abs(p[top2[0]]-p[top2[1]])

margin = sorted(list(map(lambda p : [margin(np.array(p[0:3])),p[3]], indexed_predictions)), key=lambda p : p[0])

list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[0:10]))
list(map(lambda p : predictions_on_test_data[int(p[1])], margin[0:10]))
list(map(lambda p : tweets.iloc[data_test_index[int(p[1])].toarray()[0][0]].text, margin[-10:]))
list(map(lambda p : predictions_on_test_data[int(p[1])], margin[-10:]))
import matplotlib.pyplot as plt

marginal_probs = list(map(lambda p : p[0], margin))

n, bins, patches = plt.hist(marginal_probs, 25, facecolor='blue', alpha=0.75)

plt.title('Marginal confidence histogram - All data')

plt.ylabel('Count')

plt.xlabel('Marginal probability [abs(p_positive - p_negative)]')

plt.show()
positive_test_data = list(filter(lambda row : row[0]==2, hstack((targets_test[:,None], data_test)).toarray()))

positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))

marginal_positive_probs = list(map(lambda p : abs(p[0]-p[1]), positive_probs))

n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='green', alpha=0.75)

plt.title('Marginal confidence histogram - Positive data')

plt.ylabel('Count')

plt.xlabel('Marginal probability [abs(p_positive - p_negative)]')

plt.show()
positive_test_data = list(filter(lambda row : row[0]==1, hstack((targets_test[:,None], data_test)).toarray()))

positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))

marginal_positive_probs = list(map(lambda p : abs(p[0]-p[1]), positive_probs))

n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='green', alpha=0.75)

plt.title('Marginal confidence histogram - Positive data')

plt.ylabel('Count')

plt.xlabel('Marginal probability [abs(p_positive - p_negative)]')

plt.show()
negative_test_data = list(filter(lambda row : row[0]==0, hstack((targets_test[:,None], data_test)).toarray()))

negative_probs = clf.predict_proba(list(map(lambda r : r[1:], negative_test_data)))

marginal_negative_probs = list(map(lambda p : abs(p[0]-p[1]), negative_probs))

n, bins, patches = plt.hist(marginal_negative_probs, 25, facecolor='red', alpha=0.75)

plt.title('Marginal confidence histogram - Negative data')

plt.ylabel('Count')

plt.xlabel('Marginal probability [abs(p_positive - p_negative)]')

plt.show()
import tensorflow as tf