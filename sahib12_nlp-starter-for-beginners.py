# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# XGBoost

import xgboost as xgb

from xgboost import XGBClassifier



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
#training Data

train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print('Training Data Shape',train.shape)

train.head()
# Testing data 

test = pd.read_csv('../input/nlp-getting-started/test.csv')

print('Testing data shape: ', test.shape)

test.head()

# Counting Number oF Missing values

train.isnull().sum()
#Missing values in test set

test.isnull().sum()
train['target'].value_counts()
sns.barplot(train['target'].value_counts().index,train['target'].value_counts())
# A disaster tweet

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
#not a disaster tweet

non_disaster_tweets = train[train['target']==0]['text']

non_disaster_tweets.values[1]

plt.figure(figsize=(15,10))

sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],

            orient='h')

# glance at training data

train['text'][:5]
# Applying a first round of text cleaning techniques



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train['text'].head()
text = "Are you coming , aren't you"

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')



print("Example Text: ",text)

print("------------------------------------------------------------------------------------------------")

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head()
# Not used

# # list of list 

# docs=[]

# for i in range(len(train)):

#     docs.append(train['text'][i])

    

# we are not using this step

# After preprocessing, the text format

# def combine_text(list_of_text):

#     '''Takes a list of text and combines them into one large chunk of text.'''

#     combined_text = ' '.join(list_of_text)

#     return combined_text



# train['text'] = train['text'].apply(lambda x : combine_text(x))

# test['text'] = test['text'].apply(lambda x : combine_text(x))

# train['text']

# train.head()
# Not using here

# # using dfiffrent tokenizer and TfIdf fro SKlearn

# # http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/

# from sklearn.feature_extraction.text import TfidfVectorizer



# def dummy_fun(doc):

#     return doc



# tfidf = TfidfVectorizer(

#     analyzer='word',

#     tokenizer=dummy_fun,

#     preprocessor=dummy_fun,

#     token_pattern=None)  



# # trainsforming training vector

# train_vectors=tfidf.fit_transform(docs)

# print(len(tfidf.vocabulary_)) # len of vocublary

# print(type(train_vectors))



# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text

train['text']=train['text'].apply(lambda x: combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))
# NOt used Here

# # transforming test vectors

# test_vectors = tfidf.transform(test["text"])

# print(type(test_vectors))

# print(test_vectors.shape)

# min_df and max_df 

#https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer

# ngram_range

#https://www.kaggle.com/c/avito-demand-prediction/discussion/58819

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_vectors = tfidf.fit_transform(train['text'])

test_vectors = tfidf.transform(test["text"])
clf = LogisticRegression(C=0.90,max_iter=1000,penalty='l2')

# clf = LogisticRegression(C=1.00)

# was  also best when min_df=2,max_df=5,ngram_range=(1,2)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=7, scoring="f1")

scores
clf.fit(train_vectors, train["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False)


# from sklearn.pipeline import Pipeline



# # Create first pipeline for base without reducing features.



# pipe = Pipeline([('classifier' , LogisticRegression())])





# # Create param grid.



# param_grid = [

#     {'classifier' : [LogisticRegression()],

#      'classifier__penalty' : ['l1', 'l2'],

#     'classifier__C' : np.logspace(-4, 4, 20),

#     'classifier__solver' : ['liblinear']},

    

# ]



# # Create grid search object



# clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# # Fit on data



# best_clf = clf.fit(train_vectors, train["target"])





# sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# sample_submission["target"] = best_clf.predict(test_vectors)

# sample_submission.to_csv("submission.csv", index=False)