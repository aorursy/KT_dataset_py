import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # advanced data visualization



sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.head()
train.shape
test.head()
test.shape
def find_missing_values(data_frame):    

    total = data_frame.isnull().sum().sort_values(ascending = False)

    percent = round((data_frame.isnull().sum()/data_frame.isnull().count()*100).sort_values(ascending = False), 2)

    ms = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms = ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(ms.index, ms["Percent"], alpha=0.75)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return ms
find_missing_values(train)
find_missing_values(test)
total = train.groupby(['target']).count()['id']

percent = round((train.groupby(['target']).count()['id'] / train['target'].count()) * 100, 2)

output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

output
len(train['keyword'].unique())
def count_unique(column_name):

    

    ### counts how many a certain value appears in a column and sorts it descending

    

    unique_words = train[column_name].unique()



    words = []

    counts = []



    for word in unique_words:

        if (word == word) is False: # detect NaN value

            words.append('NaN')

            counts.append(train[column_name].isnull().sum())

        else:

            words.append(word)

            counts.append(train[column_name][train[column_name] == word].count())



    words_df = pd.DataFrame(data=[words, counts], index=[column_name, 'counts'])

    words_df = words_df.T

    words_df = words_df.sort_values(by='counts', ascending=False)

    return(words_df)
unique_keywords = count_unique('keyword')

unique_keywords.head(10)
def show_ones_and_zeros(data_frame, column_name):

    

    ### adds columns with information about number of ones and zeros as well as percent of ones

    

    ones_list = []

    zeros_list = []

    percent_list = []



    for index, row in data_frame.iterrows():

        if index == 0:

            ones = train['target'][train[column_name].isnull()].sum()

            zeros = row['counts'] - ones

            ones_list.append(ones)

            zeros_list.append(zeros)

            if zeros == 0:

                percent_list.append(100)

            else:

                percent_list.append(round(ones/(zeros+ones) * 100, 2))

        else:

            ones = train['target'][train[column_name] == row[column_name]].sum()

            zeros = row['counts'] - ones

            ones_list.append(ones)

            zeros_list.append(zeros)

            if zeros == 0:

                percent_list.append(100)

            else:

                percent_list.append(round(ones/(zeros+ones) * 100, 2))



    ones_and_zeros = pd.DataFrame(data=[ones_list, zeros_list, percent_list], index=['1', '0', 'percent'])

    ones_and_zeros = ones_and_zeros.T

    keywords_df = pd.concat([data_frame, ones_and_zeros], axis=1)

    keywords_df = keywords_df.sort_values(by='percent', ascending=False)

    return(keywords_df)
show_ones_and_zeros(unique_keywords, 'keyword').head()
def create_multi_index(data_frame, column_name):

    

    ### transforms dataframe to a MultiIndex pandas series

    

    multi_index_keywords = []

    multi_index_values = []



    for index, row in data_frame.iterrows():

        multi_index_keywords.append((row[column_name], 'ones'))

        multi_index_keywords.append((row[column_name], 'zeros'))

        multi_index_values.append(row['1'])

        multi_index_values.append(row['0'])



    index = pd.MultiIndex.from_tuples(multi_index_keywords, names=[column_name, 'disaster'])



    multi_index_keywords_df = pd.Series(multi_index_values, index=index, name='value')

    return(multi_index_keywords_df)
top_10_keywords = unique_keywords.head(10)

top_10_keywords = top_10_keywords.reset_index(drop=True)

df = create_multi_index(show_ones_and_zeros(top_10_keywords, 'keyword'), 'keyword').unstack()



df['f'] = df['ones'] / (df['ones'] + df['zeros'])

df = df.sort_values(by='f', ascending=False).drop('f', axis=1)

df.plot(kind='bar', figsize=(10,10), alpha=0.75, title='Top 10 most common keywords - target variable graph').set_ylabel('Number of disaster records')
most_ones_keywords = show_ones_and_zeros(unique_keywords, 'keyword').head(10)

most_ones_keywords = most_ones_keywords.reset_index(drop=True)

df = create_multi_index(most_ones_keywords, 'keyword').unstack()



df['f'] = df['ones'] / (df['ones'] + df['zeros'])

df = df.sort_values(by='f', ascending=False).drop('f', axis=1)

df.plot(kind='bar', figsize=(10,10), alpha=0.75, title='Top 10 most disastrous keywords - target variable graph').set_ylabel('Number of disaster records')
least_ones_keywords = show_ones_and_zeros(unique_keywords, 'keyword').tail(10)

least_ones_keywords = least_ones_keywords.reset_index(drop=True)

df = create_multi_index(least_ones_keywords, 'keyword').unstack()



df['f'] = df['ones'] / (df['ones'] + df['zeros'])

df = df.sort_values(by='f').drop('f', axis=1)

df.plot(kind='bar', figsize=(10,10), alpha=0.75, title='Top 10 least disastrous keywords - target variable graph').set_ylabel('Number of disaster records')
len(train['location'].unique())
unique_locations = count_unique('location')

unique_locations.head(10)
top_10_locations = unique_locations.head(10)

top_10_locations = top_10_locations.reset_index(drop=True)

df = create_multi_index(show_ones_and_zeros(top_10_locations, 'location'), 'location').unstack()



df['f'] = df['ones'] / (df['ones'] + df['zeros'])

df = df.sort_values(by='f', ascending=False).drop('f', axis=1)

df.plot(kind='bar', figsize=(10,10), alpha=0.75, title='Top 10 most common locations - target variable graph').set_ylabel('Number of disaster records')
train['text'].apply(len).describe()
import nltk

import os

import random

import time

from collections import Counter

from nltk import word_tokenize, WordNetLemmatizer

from nltk.corpus import stopwords

from nltk import NaiveBayesClassifier, classify



stoplist = stopwords.words('english')



def preprocess(sentence):

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]



def get_features(text, setting):

    if setting=='bow':

        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}

    else:

        return {word: True for word in preprocess(text) if not word in stoplist}



def evaluate(train_set, test_set, classifier):

    # check how the classifier performs on the training and test sets

    print ('Accuracy on the training set = ' + str(round(classify.accuracy(classifier, train_set), 2)))

    print ('Accuracy of the test set = ' + str(round(classify.accuracy(classifier, test_set), 2)))

    # check which words are most informative for the classifier
train['text'][0]
preprocess(train['text'][0])
text_key_loc = []

for index, row in train.iterrows():

    if not (row['keyword'] == row['keyword'] or row['location'] == row['location']):

        text_key_loc.append(row['text'])

    elif not (row['keyword'] == row['keyword']):

        text_key_loc.append(row['text'] + ' ' + row['location'])

    elif not (row['location'] == row['location']):

        text_key_loc.append(row['text'] + ' ' + row['keyword'])

    else:

        text_key_loc.append(str(row['text']) + ' ' + str(row['location']) + ' ' + str(row['keyword']))

        



train['text_key_loc'] = text_key_loc
def train_classifier(features, samples_proportion):

    train_size = int(len(features) * samples_proportion)

    # initialise the training and test sets

    train_set, test_set = features[:train_size], features[train_size:]

    print ('Training set size = ' + str(len(train_set)) + ' tweets')

    print ('Test set size = ' + str(len(test_set)) + ' tweets')

    # train the classifier

    classifier = NaiveBayesClassifier.train(train_set)

    return train_set, test_set, classifier



SEED = 420

random.seed(SEED)

all_tweets_train = [(tweet, 1) for tweet in train['text_key_loc'][train['target'] == 1]]

all_tweets_train += [(tweet, 0) for tweet in train['text_key_loc'][train['target'] == 0]]

random.shuffle(all_tweets_train)



# extract the features

all_features = [(get_features(tweet, ''), label) for (tweet, label) in all_tweets_train]

print ('Collected ' + str(len(all_tweets_train)) + ' feature sets')



# start measuring the time

start = time.time()

# train the classifier

train_set, test_set, classifier = train_classifier(all_features, 0.8)

#stop measuring the time

end = time.time()

# evaluate its performance

evaluate(train_set, test_set, classifier)

# how long did it take?

result = round(end - start, 2)

print('Training the Naive Bayes Classifier classifier took ' + str(result) + ' seconds.')

print(classifier.show_most_informative_features())
from sklearn.naive_bayes import MultinomialNB

from nltk.classify.scikitlearn import SklearnClassifier



def train_classifier(features, samples_proportion):

    train_size = int(len(features) * samples_proportion)

    # initialise the training and test sets

    train_set, test_set = features[:train_size], features[train_size:]

    print ('Training set size = ' + str(len(train_set)) + ' tweets')

    print ('Test set size = ' + str(len(test_set)) + ' tweets')

    # train the classifier

    MNB_classifier = SklearnClassifier(MultinomialNB())

    classifier = MNB_classifier.train(train_set)

    return train_set, test_set, classifier



# start measuring the time

start = time.time()

# train the classifier

train_set, test_set, classifier = train_classifier(all_features, 0.8)

#stop measuring the time

end = time.time()

# evaluate its performance

evaluate(train_set, test_set, classifier)

# how long did it take?

result = round(end - start, 2)

print('Training the Multinominal Naive Bayes Classifier classifier took ' + str(result) + ' seconds.')  
from sklearn.naive_bayes import BernoulliNB



def train_classifier(features, samples_proportion):

    train_size = int(len(features) * samples_proportion)

    # initialise the training and test sets

    train_set, test_set = features[:train_size], features[train_size:]

    print ('Training set size = ' + str(len(train_set)) + ' tweets')

    print ('Test set size = ' + str(len(test_set)) + ' tweets')

    # train the classifier

    BNB_classifier = SklearnClassifier(BernoulliNB())

    classifier = BNB_classifier.train(train_set)

    return train_set, test_set, classifier



# start measuring the time

start = time.time()

# train the classifier

train_set, test_set, classifier = train_classifier(all_features, 0.8)

#stop measuring the time

end = time.time()

# evaluate its performance

evaluate(train_set, test_set, classifier)

# how long did it take?

result = round(end - start, 2)

print('Training the Bernoulli Naive Bayes Classifier classifier took ' + str(result) + ' seconds.')
text_key_loc = []

for index, row in test.iterrows():

    if not (row['keyword'] == row['keyword'] or row['location'] == row['location']):

        text_key_loc.append(row['text'])

    elif not (row['keyword'] == row['keyword']):

        text_key_loc.append(row['text'] + ' ' + row['location'])

    elif not (row['location'] == row['location']):

        text_key_loc.append(row['text'] + ' ' + row['keyword'])

    else:

        text_key_loc.append(str(row['text']) + ' ' + str(row['location']) + ' ' + str(row['keyword']))

        



test['text_key_loc'] = text_key_loc
sample_submission.head()
def train_classifier(features, samples_proportion):

    train_size = int(len(features) * samples_proportion)

    # initialise the training and test sets

    train_set, test_set = features[:train_size], features[train_size:]

    print ('Training set size = ' + str(len(train_set)) + ' tweets')

    print ('Test set size = ' + str(len(test_set)) + ' tweets')

    # train the classifier

    MNB_classifier = SklearnClassifier(MultinomialNB())

    classifier = MNB_classifier.train(train_set)

    return train_set, test_set, classifier



train_set, test_set, classifier = train_classifier(all_features, 0.8)



test_features = [get_features(tweet, '') for tweet in test['text_key_loc']]

print ('Collected ' + str(len(test_features)) + ' feature sets')



test['target'] = classifier.classify_many(test_features)



submission = test[['id', 'target']]

submission.head()

submission.to_csv('submission.csv', index=False)