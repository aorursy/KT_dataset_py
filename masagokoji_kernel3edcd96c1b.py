# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
# List files available

print(os.listdir("/kaggle/input/"))
#Training data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print('Training data shape: ', train.shape)

train.head()
# Testing data 

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print('Testing data shape: ', test.shape)

test.head()
#Missing values in training set

train.isnull().sum()
#Missing values in test set

test.isnull().sum()
train['target'].value_counts()
sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='rocket')
# A disaster tweet

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
#not a disaster tweet

non_disaster_tweets = train[train['target']==0]['text']

non_disaster_tweets.values[1]
sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],orient='h')
train.loc[train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
# Replacing the ambigious locations name with Standard names

train['location'].replace({'United States':'USA',

                           'New York':'USA',

                            "London":'UK',

                            "Los Angeles, CA":'USA',

                            "Washington, D.C.":'USA',

                            "California":'USA',

                             "Chicago, IL":'USA',

                             "Chicago":'USA',

                            "New York, NY":'USA',

                            "California, USA":'USA',

                            "FLorida":'USA',

                            "Nigeria":'Africa',

                            "Kenya":'Africa',

                            "Everywhere":'Worldwide',

                            "San Francisco":'USA',

                            "Florida":'USA',

                            "United Kingdom":'UK',

                            "Los Angeles":'USA',

                            "Toronto":'Canada',

                            "San Francisco, CA":'USA',

                            "NYC":'USA',

                            "Seattle":'USA',

                            "Earth":'Worldwide',

                            "Ireland":'UK',

                            "London, England":'UK',

                            "New York City":'USA',

                            "Texas":'USA',

                            "London, UK":'UK',

                            "Atlanta, GA":'USA',

                            "Mumbai":"India"},inplace=True)



sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],orient='h')
# A quick glance over the existing data

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

from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disaster Tweets',fontsize=40);
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
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train['text']

train.head()
# text preprocessing function

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

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train["target"])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

scores
# Fitting a simple Naive Bayes on Counts

clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_vectors, train["target"], cv=5, scoring="f1")

scores
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_tfidf, train["target"])
import xgboost as xgb

clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb, train_vectors, train["target"], cv=5, scoring="f1")

scores
import xgboost as xgb

clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_vectors)