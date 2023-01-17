import numpy as np
import pandas as pd
import random
import os
import string
import re

import matplotlib.pyplot as plt
import seaborn as sns
plt.set_cmap('PiYG_r')
sns.set_palette("PiYG_r")

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')
# Download stopwords
nltk.download('stopwords')
def set_seed(seed = 42):
    """For reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed()
train_raw = pd.read_csv("../input/nlp-getting-started/train.csv")
test_raw = pd.read_csv("../input/nlp-getting-started/test.csv")
train_raw.head()
train_raw.shape, test_raw.shape
train_raw.info()
# Look at the distribution of the target variable
sns.countplot(train_raw.target)
plt.title("#Disaster Tweets is Less Than Non-Disaster Tweets")
plt.show()
train_raw.target.mean()
# Randomly sample 5 disaster tweets from training data.
sample_1 = train_raw[train_raw.target == 1].sample(n=5, random_state=42)
for i in range(5):
    print(str(i+1),'.', sample_1.iloc[i].text, '\n')
# Randomly sample 5 non-disaster tweets from training data.
sample_0 = train_raw[train_raw.target == 0].sample(n=5, random_state=42)
for i in range(5):
    print(str(i+1),'.', sample_0.iloc[i].text, '\n')
text_raw = train_raw.text
target_raw = train_raw.target.values

X_train, X_test, y_train, y_test = train_test_split(text_raw, target_raw, test_size = 0.2, 
                                                    stratify = target_raw, random_state = 2020)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words (unigram) containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, disaster_status) pair to its
        frequency
    """

    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs

# create frequency dictionary
freqs = build_freqs(X_train, y_train)
len(freqs)
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. 
        loglikelihood: the log likelihood of you Naive bayes equation. 
    '''
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab) # the number of unique words in the vocabulary

    N_pos = N_neg = V_pos = V_neg = 0
    for pair in freqs.keys():
        # if the label is disaster (greater than zero)
        if pair[1] > 0:
            V_pos += 1
            N_pos += freqs[pair]

        # else, the label is not disaster
        else:
            V_neg += 1
            N_neg += freqs[pair]

    D = train_y.shape[0]  # the number of tweets

    D_pos = sum(train_y)  # the number of disaster tweets
    D_neg = D - D_pos  # # the number of non-disaster tweets

    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        # get the disaster and non-disaster frequency of the word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        # calculate the probability that each word is disaster, and non-disaster
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)


    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train)
print(logprior)
print(len(loglikelihood))
def naive_bayes_predict(tweets, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    res = []
    for tweet in tweets:
        # process the tweet to get a list of words
        word_l = process_tweet(tweet)
    
        # initialize probability to zero
        p = 0

        # add the logprior
        p += logprior

        for word in word_l:

            # check if the word exists in the loglikelihood dictionary
            if word in loglikelihood:
                # add the log likelihood of that word to the probability
                p += loglikelihood[word]
    
        if p > 0:
            res.append(1)
        else:
            res.append(0)


    return res
train_preds = naive_bayes_predict(X_train, logprior, loglikelihood)
test_preds = naive_bayes_predict(X_test, logprior, loglikelihood)
print('TRAINING SET: accuracy score: {}, F1 score: {}'.format(accuracy_score(y_train, train_preds), 
                                                              f1_score(y_train, train_preds)))
print('TEST SET: accuracy score: {}, F1 score: {}'.format(accuracy_score(y_test, test_preds), 
                                                         f1_score(y_test, test_preds)))
# Predict test data
test = test_raw.text
test_res = naive_bayes_predict(test, logprior, loglikelihood)
res = pd.DataFrame({'id': test_raw.id, 'target': test_res})
res.head()
# res.to_csv("naive_bayes_submission.csv", header=True, index=False)

