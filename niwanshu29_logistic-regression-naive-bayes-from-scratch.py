# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import twitter_samples

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
len(all_positive_tweets)
# Select any positive tweet
tweet = all_positive_tweets[2277]
tweet
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    #tokenizer
    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
    lemmatizer = WordNetLemmatizer()
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    tweets_lemma = []
    for tweet in tweet_tokens:
        if (tweet not in stopwords_english) and (tweet not in string.punctuation):
            stem_word = stemmer.stem(tweet)
            lemma_word = lemmatizer.lemmatize(tweet)
            tweets_clean.append(stem_word)
            tweets_lemma.append(lemma_word)
    return tweets_clean,tweets_lemma
def build_freqs(tweets,ys):
    freqs_stem = {}
    freqs_lemma = {}
    ylist = np.squeeze(ys).tolist()
    for y,tweet in zip(ys,tweets):
        tweets,tweets_lemma = process_tweet(tweet)
        for word in tweets:
            pair = (word,y)
            if pair in freqs_stem:
                freqs_stem[pair] += 1
            else:
                freqs_stem[pair] = 1
        for word in tweets_lemma:
            pair = (word,y)
            if pair in freqs_lemma:
                freqs_lemma[pair] += 1
            else:
                freqs_lemma[pair] = 1
    return freqs_stem,freqs_lemma
build_freqs([all_positive_tweets[121],all_negative_tweets[121]],np.array([1,0]))
tweets = all_positive_tweets + all_negative_tweets
labels = np.append(np.ones(len(all_positive_tweets)) , np.zeros(len(all_negative_tweets)))
len(labels)
#type(labels[0])
freqs,freqs_lemma = build_freqs(tweets,labels)
# select some words to appear in the report. we will assume that each word is unique (i.e. no duplicates)
keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']
data = []
for word in keys:
    pos = 0
    neg = 0
    if (word,1) in freqs:
        pos = freqs[(word,1)]
    if (word,0) in freqs:
        neg = freqs[(word,0)]
    data.append([word,pos,neg])
data
    
fig,ax = plt.subplots(figsize = (8,8))

x = np.log([x[1] + 1 for x in data])
y = np.log([x[2] + 1 for x in data])
ax.scatter(x,y)
plt.xlabel('Log Positive Count')
plt.ylabel('Log Negative Count')

for i in range(len(data)):
    ax.annotate(data[i][0],(x[i],y[i]) , fontsize = 12)

ax.plot([0, 9], [0, 9], color = 'red') # Plot the red line that divides the 2 areas.
plt.show()
test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]

train_set = train_pos+train_neg
test_set = test_pos+test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))
freqs_stem , freqs_lemma = build_freqs(train_set,np.squeeze(train_y).tolist())
def extract_features(tweet,freqs_stem,freqs_lemma):
    word_stem,word_lemma = process_tweet(tweet)
    x_stem = np.zeros((1,3))
    x_lemma = np.zeros((1,3))
    x_stem[0,0] = 1
    x_lemma[0,0] = 1
    
    for word in word_stem:
        if (word,1) in freqs_stem:
            x_stem[0,1] += freqs_stem[(word,1)]
        if (word,0) in freqs_stem:
            x_stem[0,2] += freqs_stem[(word,0)]
            
    for word in word_lemma:
        if (word,1) in freqs_lemma:
            x_lemma[0,1] += freqs_lemma[(word,1)]
        if (word,0) in freqs_lemma:
            x_lemma[0,2] += freqs_lemma[(word,0)]
        
    return x_stem,x_lemma
extract_features(train_set[1],freqs_stem,freqs_lemma)
def sigmoid(z):
    return 1/(1+np.exp(-z))
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def gradient_descent(x, y, theta, alpha, num_iters):
    m = len(x)
    costs = []
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = (-1/m)*(np.dot(np.transpose(y),np.log(h)) + np.dot(np.transpose(1-y),(np.log(1-h))))
        costs.append(J)
        # update the weights theta
        theta = theta - (alpha/m)*(np.dot(np.transpose(x),(h-y))) 
        
    J = float(J)
    return J, theta,costs
# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta,costs = gradient_descent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")
# Extracting Features for Training Data
X_stem = np.zeros((len(train_set),3))
X_lemma = np.zeros((len(train_set),3))
for i in range(len(train_set)):
    X_stem[i,:],X_lemma[i,:] = extract_features(train_set[i],freqs_stem,freqs_lemma)
    
Y = train_y

# Apply gradient descent
J, theta_stem,costs_stem = gradient_descent(X_stem, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training(stemming) is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta_stem)]}")

# Apply gradient descent
J, theta_lemma,costs_lemma = gradient_descent(X_lemma, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training(Lemmatization) is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta_lemma)]}")
costs_lemma = np.array(costs_lemma).reshape(-1,1)
X = np.array([i for i in range(1,1501)]).reshape(-1,1)
costs_stem = np.array(costs_stem).reshape(-1,1)
plt.plot(X,costs_lemma)
plt.plot(X,costs_stem)
plt.legend(["Lemmatization","Stemming"])
plt.show()
def predict_tweet(tweet,theta,freqs):
    x_stem,x_lemma = extract_features(tweet,freqs_stem,freqs_lemma)
    y_pred = sigmoid(np.dot(x_stem,theta))
    return y_pred

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet,theta_stem, freqs_stem)))
def test_logistic_regression(test_x,test_y,freqs,theta):
    y_hat = []
    
    for tweet in test_x:
        y_pred = predict_tweet(tweet,theta,freqs)
        
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
        
    accuracy = (y_hat== np.squeeze(test_y)).sum()/len(test_x)
    return accuracy
tmp_accuracy = test_logistic_regression(test_set, test_y, freqs_stem, theta_stem)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    #tokenizer
    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)
    #lemmatizer = WordNetLemmatizer()
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    #tweets_lemma = []
    for tweet in tweet_tokens:
        if (tweet not in stopwords_english) and (tweet not in string.punctuation):
            stem_word = stemmer.stem(tweet)
            #lemma_word = lemmatizer.lemmatize(tweet)
            tweets_clean.append(stem_word)
            #tweets_lemma.append(lemma_word)
    return tweets_clean
def count_tweet(tweets,label):
    freqs = {}
    for y,tweet in zip(label,tweets):
        for word in process_tweet(tweet):
            if (word,y) in freqs:
                freqs[(word,y)] += 1
            else:
                freqs[(word,y)] = 1
    return freqs
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
count_tweet(tweets, ys)    
train_y = np.squeeze(train_y).tolist()
freqs = count_tweet(train_set,train_y)
def lookup(freqs,word,label):
    if (word,label) in freqs:
        return freqs[(word,label)]
    return 0

def train_naive_bayes(freqs , train_x, train_y):
    loglikelihood = {}
    logprior = 0
    # Unique Words
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    
    D = len(train_y) #Number of Documents
    D_pos = len(list(filter(lambda x:x>0,train_y)))
    D_neg = len(list(filter(lambda x:x<=0,train_y)))
    
    logprior = np.log(D_pos) - np.log(D_neg)
    
    for word in vocab:
        freqs_pos = lookup(freqs,word,1)
        freqs_neg = lookup(freqs,word,0)
        p_w_pos = (freqs_pos + 1)/(N_pos + V)     #Laplacian Smoothing, to avoid division by zero
        p_w_neg = (freqs_neg + 1)/(N_neg + V)
        
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior,loglikelihood
logprior, loglikelihood = train_naive_bayes(freqs, train_set, train_y)
print(logprior)
print(len(loglikelihood))
def naive_bayes_predict(tweet,logprior,loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p

my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)
def test_naive_bayes(test_x,test_y,logprior,loglikelihood):
    y_hat = []
    for tweet in test_x:
        y_pred = naive_bayes_predict(tweet,logprior,loglikelihood)
        
        if y_pred > 0:
            y_hat.append(1)
        else:
            y_hat.append(0)
            
    error = np.abs(np.sum(y_hat) - np.sum(test_y))/len(test_y)
    
    accuracy = 1 - error
    
    return accuracy
        
print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_set, test_y, logprior, loglikelihood)))
