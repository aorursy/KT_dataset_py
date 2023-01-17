# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import string
import time
import gc
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.linear_model import *
from gensim.models import *
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib as plt
plt.rcParams["figure.figsize"] = (20,10)
import warnings
warnings.filterwarnings('ignore')
%pylab inline
X_train = pd.read_csv("../input/train.csv", encoding='latin1')
X_test = pd.read_csv("../input/test.csv", encoding='latin1')
X_train['Sentiment'].value_counts()
X_train.head(5)
X_test.head(5)
X_train.shape
X_test.shape
def clear_tweets(s):
    tok = WordPunctTokenizer()
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nickname = r'@[A-Za-z0-9]+'
    url = r'https?://[A-Za-z0-9./]+'
    hashtag = r'#'
    pattern = r'|'.join((nickname, url, hashtag))
    s = BeautifulSoup(s).get_text()
    s = re.sub(pattern, '', s)
    s = re.sub('[^a-zA-Z]', ' ', s)
    s = s.lower()
    tockens = tok.tokenize(s)
    stems = []
    for t in tockens:
        #stems.append(porter.stem(t))
        #stems.append(lancaster.stem(t))
        stems.append(wordnet_lemmatizer.lemmatize(t, pos="v"))
    return (' '.join(stems)).strip()
stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
emoticons = emoticons_happy.union(emoticons_sad)

def clear_tweets(tweet):
    #removing vibes like retweet, nicname, hastags and etc.
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    #tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []    
    for word in tweet_tokens:
        if (#word not in stopwords_english and
              #word not in emoticons and
                word not in string.punctuation):
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            #stem_word = stemmer.lemmatize(word, pos='v') # stemming word
            tweets_clean.append(stem_word)
    
    return (' '.join(tweets_clean)).strip()
tweet = 'RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning https://chapagain.com.np'
clear_tweets(tweet)
list(X_train['SentimentText'].head(5).values)
clear_tweets(X_train['SentimentText'][3])
%%time

X_train_processed = []
count = X_train.shape[0]
for i, t in enumerate(X_train['SentimentText'].values):
    X_train_processed.append(clear_tweets(t))
    if (i+1)%10000==0:
        print('Processed:', i+1, '\n', 'of', count)
y = X_train['Sentiment'].values
list(X_train['SentimentText'].head(10).values)
X_train_processed[:10]
%%time

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train_processed, y, test_size=.25)

vectorizer = TfidfVectorizer(input='content',
                             encoding='latin1',
                             decode_error='strict',
                             strip_accents=None,
                             lowercase=True,
                             preprocessor=None,
                             tokenizer=None,
                             analyzer='word',
                             #stopwords didn't work
                             stop_words=None,
                             #token_pattern='(?u)\b\w\w+\b',
                             ngram_range=(1, 5),
                             max_df=.9,
                             min_df=3,
                             max_features=100000,
                             vocabulary=None,
                             binary=False,
                             dtype='float64',
                             norm='l2',
                             use_idf=True,
                             smooth_idf=True,
                             sublinear_tf=False)

X_train_cv = vectorizer.fit_transform(X_train_cv)
X_test_cv = vectorizer.transform(X_test_cv)
%%time

LR = LogisticRegression(penalty='l2'
                        , dual=False
                        , tol=0.0001
                        , C=1.0
                        , fit_intercept=True
                        , intercept_scaling=1
                        , class_weight=None
                        , random_state=0
                        , solver='liblinear'
                        , max_iter=100
                        , multi_class='ovr'
                        , verbose=0
                        , warm_start=False
                        , n_jobs=-1)

LR.fit(X_train_cv, y_train_cv)
predict = LR.predict(X_test_cv)
print(f1_score(y_test_cv, predict))
i = 100
for n, t in enumerate(X_test_processed[:i]):
    if y_test_cv[n-1]!=predict[n-1]:
        print(n,y_test_cv[n-1], t)
%%time

X_test_processed = []
count = X_test.shape[0]
for i, t in enumerate(X_test['SentimentText'].values):
    X_test_processed.append(clear_tweets(t))
    if (i+1)%10000==0:
        print('Processed:', i+1, '\n', 'of', count)
predict = LR.predict(vectorizer.transform(X_test_processed))
output = pd.DataFrame( data={"ItemID":X_test["ItemID"], "Sentiment":predict} )
output.to_csv("Twitter_result.csv", index=False, quoting=3 )
# G
#%%time
#%env JOBLIB_TEMP_FOLDER=/tmp
#
##using gpu for grid search cv
#
#params = {
#    'penalty': ['l1', 'l2'],
#    'max_iter': [100, 150, 250, 500],
#    'tol': [.0001, .001, .01, 1.0],
#    'C': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
#}
#
#cv = GridSearchCV(estimator=LR, param_grid=params, n_jobs=-1, cv=5, verbose=1)
#cv.fit(X=X_train_cv, y=y_train_cv)

#print(cv.best_params_)
#print(f1_score(y_test_cv, cv.best_estimator_.predict(X_test_cv)))























