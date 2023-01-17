import pandas as pd

# from pandas import options

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

from matplotlib.style import use

use('ggplot')

# options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv('../input/winemag-data_first150k.csv')

df.head(5)
# df['description'].apply(func=lambda text: len(word_tokenize(text))).hist()

# plt.title('Number of words used to describe a wine.')

# plt.xlabel('Number of words in description.')

# plt.ylabel('Number of reviews.')

# plt.show()
df = df.dropna(subset=['price', 'points'])

df[['price', 'points']].hist(layout=(2,1))

plt.show()
plt.scatter(df['points'], df['price'])

plt.xlabel('Points')

plt.ylabel('Price')

plt.show()
df = df.dropna(subset=['description'])  # drop all NaNs



df_sorted = df.sort_values(by='points', ascending=True)  # sort by points



num_of_wines = df_sorted.shape[0]  # number of wines

worst = df_sorted.head(int(0.25*num_of_wines))  # 25 % of worst wines listed

best = df_sorted.tail(int(0.25*num_of_wines))  # 25 % of best wines listed
plt.hist(df['points'], color='grey', label='All')

plt.hist(worst['points'], color='blue', label='Worst')

plt.hist(best['points'], color='red', label='Best')

plt.legend()

plt.show()
from nltk.tokenize import word_tokenize

from nltk import FreqDist, NaiveBayesClassifier

from random import shuffle
worst['words'] = worst['description'].apply(func=lambda text: word_tokenize(text.lower()))

best['words'] = best['description'].apply(func=lambda text: word_tokenize(text.lower()))

worst = worst.dropna(subset=['words'])  # drop all NaNs

best = best.dropna(subset=['words'])  # drop all NaNs
all_words = []  # initialize list of all words

# add all words from 'worst' dataset

for description in worst['words'].values:

    for word in description:

        all_words.append(word)

# add all words from 'best' dataset

for description in best['words'].values:

    for word in description:

        all_words.append(word)

all_words = FreqDist(all_words)  # make FreqList

words_features = list(all_words.keys())[:3000]  # select 3000 most frequent words as words features
def find_features(doc):

    """Function for making features out of the text"""

    words = set(doc)  # set of words in description

    features = {}  # feature dictionary

    for w in words_features:  # check if any feature word is presented

        features[w] = bool(w in words)  # write to feature vector

    return features  # return feature vector
featureset = ([(find_features(description), 'worst') for description in worst['words']] +

              [(find_features(description), 'best') for description in best['words']])

shuffle(featureset)  # randomly shuffle dataset
classifier = NaiveBayesClassifier.train(labeled_featuresets=featureset)
classifier.show_most_informative_features(50)