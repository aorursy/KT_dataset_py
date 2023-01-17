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
train=pd.read_csv(r"/kaggle/input/nlp-getting-started/train.csv")

test=pd.read_csv(r"/kaggle/input/nlp-getting-started/test.csv")

sample_submission=pd.read_csv(r"/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
train.shape
train.isnull().sum()
test.isnull().sum()
train['target'].value_counts()
import seaborn as sns
sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='rocket')
train[['text','target']].head(20)
#lets look separately how non disaster tweet(0) look like



for i in range(10):

    res=train[train['target']==0]['text'].values[i] 

    print(res)
#lets look separately how  disaster tweet(1) look like



for i in range(10):

    res=train[train['target']==1]['text'].values[i] 

    print(res)
pd.DataFrame(train['keyword'].value_counts()[:20])
sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],orient='h')
pd.DataFrame(train['location'].value_counts()).head(20)
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



sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],

            orient='h')
import re
# Applying a first round of text cleaning techniques since we have seen the text is having noise



def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train['text'].head()
import nltk
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
from nltk.corpus import stopwords
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head()
from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

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
from sklearn.feature_extraction.text import CountVectorizer
#  count vectorizer

count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])
from sklearn.feature_extraction.text import TfidfVectorizer
#Tf-Idf

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
#fitting the model to countvectorizer

clf=LogisticRegression(C=1.0)

scores=cross_val_score(clf,train_vectors,train['target'],cv=7,scoring="f1")

scores
scores.mean()
clf.fit(train_vectors,train['target'])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

scores
scores.mean()
from sklearn.naive_bayes import MultinomialNB

# Fitting a simple Naive Bayes on Counts

clf_NB=MultinomialNB()

scores=cross_val_score(clf_NB,train_vectors,train['target'],cv=5,scoring="f1")

scores
scores.mean()
clf_NB.fit(train_vectors,train["target"])
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores =cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
scores.mean()
clf_NB_TFIDF.fit(train_tfidf, train["target"])
sample_submission['target']=clf_NB.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)