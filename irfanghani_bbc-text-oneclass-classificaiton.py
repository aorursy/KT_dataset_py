# Load packages

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.svm import OneClassSVM

from sklearn.utils import shuffle

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import classification_report 

from nltk.corpus import stopwords

import statistics

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.stem.porter import PorterStemmer

import string

import spacy

from spacy.lang.en import English

import numpy as np

import scipy.stats as stats

import math

spacy.load('en')

parser = English()
# load dataset

bbc_df = pd.read_csv('../input/bbc-text.csv')
bbc_df.head(10)
def wc_count(text):

    if isinstance(text,str):

        return len(text.split())

    else:

        return 0

bbc_df['word_count'] = bbc_df['text'].apply(wc_count)
bbc_df.columns
print('Size of corpus',len(bbc_df['word_count']))

print('max word count',max(bbc_df['word_count']))

print('min word count',min(bbc_df['word_count']))
# Category Distribution

bbc_df.groupby('category').count()
# Plotting the normal distribution

std = statistics.stdev(bbc_df['word_count'])

mean = statistics.mean(bbc_df['word_count'])

variance = statistics.variance(bbc_df['word_count'])

x_min = min(bbc_df['word_count'])

x_max = max(bbc_df['word_count'])

x= bbc_df['word_count']



mu = mean

variance = 1

sigma = math.sqrt(variance)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma))

plt.show()
bbc_df.sort_values('word_count',inplace = True)
bbc_df['word_count'].tail(15)
bbc_df['word_count'].head(15)
bbc_df = bbc_df.iloc[:-15]
# Plotting the Normal Distribution after removing the outliers



std = statistics.stdev(bbc_df['word_count'])

mean = statistics.mean(bbc_df['word_count'])

variance = statistics.variance(bbc_df['word_count'])

x_min = min(bbc_df['word_count'])

x_max = max(bbc_df['word_count'])

x= bbc_df['word_count']



mu = mean

variance = 1

sigma = math.sqrt(variance)

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma))

plt.show()

# Round to the nearest 100th value

import math

def roundup(x):

    return int(math.ceil(x / 100.0)) * 100



bbc_df['word_count'] = bbc_df['word_count'].apply(roundup)



# plotting the histogram

score_india = bbc_df['word_count']

legend = 'Word Count'

plt.hist(score_india, color='green')

plt.xlabel("Article Size")

plt.ylabel("Frequency")

plt.legend(legend)

plt.xticks(range(0, 1500, 200))

plt.yticks(range(0, 1000, 100))

plt.title('Word count for articles in BBC')

plt.figure(figsize=(3,7))

plt.show()
bbc_df.shape
bbc_df.info()
bbc_df['category'].unique()
bbc_df['category'].value_counts()
sns.countplot(bbc_df['category'])