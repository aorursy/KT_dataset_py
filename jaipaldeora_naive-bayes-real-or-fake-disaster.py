# necessary imports 

import numpy as np

import pandas as pd 

import re

import string

import matplotlib.pyplot as plt

import seaborn as sns



import nltk

from nltk.tokenize import TweetTokenizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords



from sklearn.metrics import f1_score
# Load the data

df = pd.read_csv('../input/nlp-getting-started/train.csv',index_col='id')

df.head()
df.info()
# drop unnecessary cols

df.drop(['keyword','location'],axis=1,inplace=True)
def process_tweet(tweet):

    stopwords_english = stopwords.words('english')

    stemmer = PorterStemmer()

    

    tweet = re.sub(r'\$\w*','',tweet) # remove tickers

    tweet = re.sub(r'^RT[\s]+','',tweet) # remove retweet text RT  if any

    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet) # remove hyperlinks

    tweet = re.sub(r'#','',tweet) # remove #tag sign

    

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    

    tokens = tokenizer.tokenize(tweet)

    

    tweets_clean = []

    

    for word in tokens:

        if (word not in stopwords_english) and (word not in string.punctuation): # remove punctuation and stopwords

            stem = stemmer.stem(word) # apply stemming

            tweets_clean.append(stem)



    return tweets_clean
def build_freqs(result, tweets, targets):

    

    for target, tweet in zip(targets,tweets):

        for word in process_tweet(tweet):

            pair = (word,target)

            result[pair] = result.get(pair,0) + 1

            

    return result
freqs = build_freqs({}, tweets=df['text'].values, targets=df['target'].values)
def lookup(freqs,word,target):

    return freqs.get((word,target), 0)
def train(freqs, x,y):

    loglikelihood = {}

    logprior = 0

    

    vocab = set([k[0] for k in freqs.keys()])

    V = len(vocab)

    

    N_pos = N_neg = 0

    for pair in freqs.keys():

        if pair[1] > 0:

            N_pos += freqs[pair]

        else:

            N_neg += freqs[pair]

    

    D = len(y)

    

    D_pos = np.sum(y > 0)

    D_neg = D - D_pos

    

    logprior = np.log(D_pos) - np.log(D_neg)

    

    for word in vocab:

        freq_pos = lookup(freqs,word,1)

        freq_neg = lookup(freqs,word,0)

        

        p_w_pos = (freq_pos + 1)/(N_pos + V)

        p_w_neg = (freq_neg + 1)/(N_neg + V)

        

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

        

    return logprior, loglikelihood



logprior, loglikelihood = train(freqs,df['text'].values, df['target'].values)
def predict(tweet, logprior, loglikelihood):

    word_l = process_tweet(tweet)

    

    p = 0

    

    p += logprior

    

    for word in word_l:

        if word in loglikelihood:

            p += loglikelihood[word]

    

    return p
def test(x,y,logprior,loglikelihood):

    accuracy = 0

    

    y_hats = []

    

    for tweet in x:

        if predict(tweet, logprior,loglikelihood) > 0:

            y_hats_i = 1

        else:

            y_hats_i = 0

            

        y_hats.append(y_hats_i)

    

    error = np.abs(y - y_hats).sum() / len(x)

    

    accuracy = 1 - error 



    return accuracy, f1_score(y,y_hats), y_hats
# Test on Training Set

accuracy, f1, preds = test(df['text'].values, df['target'].values, logprior, loglikelihood)
print('Accuracy: ',accuracy)

print('F1-score: ',f1)
# load test set



tf = pd.read_csv('../input/nlp-getting-started/test.csv',index_col='id')

x = tf['text'].values

y = np.ones((len(x),1))



_, _, preds = test(x,y,logprior, loglikelihood)



sub = pd.DataFrame({'id': tf.index.values, 'target': preds})

sub.head()
sub.to_csv('submission.csv',index=False)