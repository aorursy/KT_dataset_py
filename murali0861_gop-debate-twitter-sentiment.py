# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Sentiment = pd.read_csv("../input/first-gop-debate-twitter-sentiment/Sentiment.csv")
print(Sentiment.shape)

Sentiment.info()
Sentiment.head()
data = Sentiment[['text', 'sentiment']]

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=100)
print('Train data Shape:', train_data.shape)

print('Test data Shape:', test_data.shape)
train_data['sentiment'].value_counts().plot(kind='barh')
test_data['sentiment'].value_counts().plot(kind='barh')
train_data = train_data[train_data.sentiment != 'Neutral']

test_data = test_data[test_data.sentiment != 'Neutral']



test_data_negative_tweets_count = test_data[test_data.sentiment == 'Negative'].shape[0]

test_data_positive_tweets_count = test_data[test_data.sentiment == 'Positive'].shape[0]
print("negative tweets count: ", test_data_negative_tweets_count)

print("positive tweets count: ", test_data_positive_tweets_count)
test_data['sentiment'].value_counts().plot(kind='barh')
def wordcloud_draw(data):

    words = ' '.join(data)

    cleaned_words = [word for word in words.split(' ') if 'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith('#') and 'RT' != word]

    cleaned_word = " ".join(cleaned_words)

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=2500, height=2500).generate(cleaned_word)

    plt.figure(1, figsize=(30, 30))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show
wordcloud_draw(train_data[train_data.sentiment == 'Negative'].text)
wordcloud_draw(train_data[train_data.sentiment == 'Positive'].text)
stopwords_set = set(stopwords.words("english"))
def prepare_data(data):

    tweets = []

    for index, series in data.iterrows():

        text, sentiment = series.text.lower(), series.sentiment

        words = text.split()

        words = [word for word in words if len(word) > 3 and not word in stopwords_set and  'http' not in word and not word.startswith('@') and not word.startswith('#') and not word.startswith('#') and 'RT' != word]

        tweets.append((words, sentiment))

    return tweets
train_data = prepare_data(train_data)

test_data = prepare_data(test_data)
train_data[0:5]
def get_words(tweets):

    all_words = []

    for (words, sentiment) in tweets:

        all_words.extend(words)

    return all_words



def get_word_features(tweets):

    words = get_words(tweets)

    words_by_count = nltk.FreqDist(words)

    return words_by_count.keys()

    
word_features = get_word_features(train_data)
word_features = set(word_features)
train_data[0][0]

features = {}

for word in train_data[0][0]:

    features['contains(%s)' %word] = (word in word_features)



features
def prepare_word_vc(document):

    for word in document:

        features['contains(%s)' %word] = (word in word_features) 

    return features
train_data = nltk.classify.apply_features(prepare_word_vc, train_data)
train_data
classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
for x, y in test_data[0:1]:

    print(prepare_word_vc(x))
negative_count, positive_count = 0, 0



sentiment_result = {'Negative': 0, 'Positive': 0}



for words, sentiment in test_data:

    res = classifier.classify(prepare_word_vc(words))

    

    if sentiment == res:

        sentiment_result[res] += 1
sentiment_result
print('[Negative]: %s/%s '  % (sentiment_result['Negative'], test_data_negative_tweets_count))

print('[Positive]: %s/%s '  % (sentiment_result['Positive'], test_data_positive_tweets_count))