import numpy as np 

import pandas as pd 



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

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import cufflinks as cf #importing plotly and cufflinks in offline mode  

import plotly.offline  

cf.go_offline()  

cf.set_config_file(offline=False, world_readable=True)
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
test_df.head()
print(train_df.shape)

print(test_df.shape)
from pandas_profiling import ProfileReport

profile = ProfileReport(train_df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile
location = train_df['location'].value_counts()[:20]

location.iplot(kind='bar', xTitle='Number of Location', title="Location's In The Given Data Set")
target = train_df['target'].value_counts()

target.iplot(kind='bar', title="Disaster or Not Distribution")
# A disaster tweet

disaster_tweets = train_df[train_df['target']==1]['text']

disaster_tweets.values[1]
# A Non disaster tweet

disaster_tweets = train_df[train_df['target']==0]['text']

disaster_tweets.values[1]
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
train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))

test_df['text'] = test_df['text'].apply(lambda x : clean_text(x))
train_df['text'].head()
test_df['text'].head()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train_df['text'] = train_df['text'].apply(lambda x: tokenizer.tokenize(x))

test_df['text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(x))
train_df['text'].head()
test_df['text'].head()
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words



train_df['text'] = train_df['text'].apply(lambda x : remove_stopwords(x))

test_df['text'] = test_df['text'].apply(lambda x : remove_stopwords(x))
train_df['text'].head()
test_df['text'].head()
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train_df['text'] = train_df['text'].apply(lambda x : combine_text(x))

test_df['text'] = test_df['text'].apply(lambda x : combine_text(x))
train_df.head()
test_df.head()
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
count_vectorizer = CountVectorizer()

train_df_vectors = count_vectorizer.fit_transform(train_df['text'])

test_df_vectors = count_vectorizer.transform(test_df["text"])



## Keeping only non-zero elements to preserve space 

print(train_df_vectors[0].todense())
# TFIDF Features (Term Frequency-Inverse Document Frequency, or TF-IDF for short)



tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_df_tfidf = tfidf.fit_transform(train_df['text'])

test_df_tfidf = tfidf.transform(test_df["text"])
# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_df_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_df_vectors, train_df["target"])
# Fitting a simple Logistic Regression on TFIDF



clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_df_tfidf, train_df["target"], cv=5, scoring="f1")

scores
# Fitting a simple Naive Bayes on Counts



clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_df_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf_NB.fit(train_df_vectors, train_df["target"])
# Fitting a simple Naive Bayes on TFIDF



clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_df_tfidf, train_df["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_df_tfidf, train_df["target"])
import xgboost as xgb

clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb, train_df_vectors, train_df["target"], cv=5, scoring="f1")

scores
import xgboost as xgb

clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb_TFIDF, train_df_tfidf, train_df["target"], cv=5, scoring="f1")

scores
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_df_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_df_vectors=test_df_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_df_vectors)