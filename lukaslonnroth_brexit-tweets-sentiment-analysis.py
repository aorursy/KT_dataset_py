import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt
sentiment = pd.read_csv('../input/first-gop-debate-twitter-sentiment/Sentiment.csv')

training_data = sentiment[['sentiment', 'text']]

training_data = training_data[training_data.sentiment != "Neutral"]

training_data = training_data[:100]
tweets = []

stopwords_set = set(stopwords.words("english"))



for index, row in training_data.iterrows():

    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]

    words_cleaned = [word for word in words_filtered

        if 'http' not in word

        and not word.startswith('@')

        and not word.startswith('#')

        and word != 'RT']

    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]

    tweets.append((words_without_stopwords, row.sentiment))
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

        features['contains(%s)' % word] = (word in document_words)

    return features

featuresets = nltk.classify.apply_features(extract_features,tweets)

len(featuresets)
training_set = featuresets[50:]

test_set = featuresets[:50]
naive_classifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(naive_classifier, test_set))
brexit_data = pd.read_csv('../input/brexit-tweets/brexit-26-march.csv', header=None)

brexit_data.head()
pos_tweets = []

neg_tweets = []

pos_cnt = 0

neg_cnt = 0
smaller_data_with_stopwords = brexit_data[:100]



for obj in smaller_data_with_stopwords[1]: 

    res =  naive_classifier.classify(extract_features(obj.split()))

    if(res == 'Negative'): 

        neg_tweets.append(obj)

        neg_cnt += 1

    elif(res == 'Positive'): 

        pos_tweets.append(obj)

        pos_cnt += 1
print('positive tweets: %s' %pos_cnt)

print('negative tweets: %s' %neg_cnt)
for index, row in brexit_data.iterrows():

    words_filtered = [e.lower() for e in row[1].split() if len(e) >= 3]

    words_cleaned = [word for word in words_filtered

        if 'http' not in word

        and not word.startswith('@')

        and not word.startswith('#')

        and word != 'RT']

    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]

    row[1] = ' '.join(words_without_stopwords)
brexit_data.head()
def classify_data(data, classifier):

    pos_tweets = []

    neg_tweets = []

    pos_cnt = 0

    neg_cnt = 0

    data.insert(2, 3, '')



    for index, row in data.iterrows():

        obj = row[1]

        res =  classifier.classify(extract_features(obj.split()))

        row[3] = res

        if(res == 'Negative'): 

            neg_tweets.append(obj)

            neg_cnt += 1

        elif(res == 'Positive'): 

            pos_tweets.append(obj)

            pos_cnt += 1

    return pos_tweets, neg_tweets, pos_cnt, neg_cnt, data
data = brexit_data[:2000]

pos_tweets, neg_tweets, pos_cnt, neg_cnt, classified_data = classify_data(data, naive_classifier)
classified_data.to_csv('classified_brexit_tweets.csv', index = False)
print('Positive tweets: %s' %pos_cnt)

print('Negative tweets: %s' %neg_cnt)
wordcloud = WordCloud(background_color="white", width=2500, height=2000).generate(' '.join(pos_tweets))

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Positive words in tweets')

plt.show()
wordcloud = WordCloud(width=2500, height=2000).generate(' '.join(neg_tweets))

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Negative words in tweets')

plt.show()
dtree_classifier = nltk.classify.DecisionTreeClassifier.train(training_set)
print(nltk.classify.accuracy(dtree_classifier, test_set))
data = brexit_data[:1000]

pos_tweets, neg_tweets, pos_cnt, neg_cnt, dtree_classified_data = classify_data(data, dtree_classifier)
print('Positive tweets: %s' %pos_cnt)

print('Negative tweets: %s' %neg_cnt)
wordcloud = WordCloud(background_color="white", width=2500, height=2000).generate(' '.join(pos_tweets))

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Positive words in tweets')

plt.show()
wordcloud = WordCloud(width=2500, height=2000).generate(' '.join(neg_tweets))

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Negative words in tweets')

plt.show()