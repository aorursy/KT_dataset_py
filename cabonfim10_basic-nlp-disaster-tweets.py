# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.probability import FreqDist



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression



from sklearn import metrics



import string

import re



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
train_df.head()
train_df.info()
train_df.isnull().sum()
test_df.head()
test_df.info()
test_df.isnull().sum()
print(round(train_df.target.value_counts(normalize=True)*100, 2))
sns.countplot(train_df.target)
# Analyzing the frequency

FreqDist([w.lower() for w in (sum(train_df['text'].apply(lambda x: word_tokenize(x)), []))])
# Creating a copy for clean and keep the original

train_df['text_cleaned'] = train_df.text.copy()
# Functions for clean



# Remove punctuation

def remove_punct(text):

    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct



# Lemmatization

def word_lemmat(text):

    lemmatizer = WordNetLemmatizer()

    lemmat_text = [lemmatizer.lemmatize(i) for i in text]

    return lemmat_text



# Stemming

def word_stemmer(text):

    stemmer = PorterStemmer()

    stem_text = [stemmer.stem(i) for i in text]

    return stem_text

    

# Remove stop words

def remove_stopw(text):

    stop_w = set(stopwords.words("english"))

    words = " ".join([w for w in text if w not in stop_w])

    return words
# Apply the functions



# Remove special caracteres

train_df['text_cleaned'] = train_df['text_cleaned'].str.replace('[^A-Za-z0-9+]', ' ')



train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: remove_punct(x.lower().strip()))

train_df['text_cleaned'] = train_df['text_cleaned'].apply(word_tokenize)

train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: word_lemmat(x))

train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: word_stemmer(x))

train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: remove_stopw(x))
# Apply the functions



# Remove special caracteres

test_df['text'] = test_df['text'].str.replace('[^A-Za-z0-9+]', ' ')



test_df['text'] = test_df['text'].apply(lambda x: remove_punct(x.lower().strip()))

test_df['text'] = test_df['text'].apply(word_tokenize)

test_df['text'] = test_df['text'].apply(lambda x: word_lemmat(x))

test_df['text'] = test_df['text'].apply(lambda x: word_stemmer(x))

test_df['text'] = test_df['text'].apply(lambda x: remove_stopw(x))
# Define X and y

train_df[['target','text_cleaned']].head()
X = train_df['text_cleaned']

y = train_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
nb = MultinomialNB()
vector = CountVectorizer()

vector.fit(X_train)
# Transform training data

X_train_doc = vector.transform(X_train)
# X_train_doc = vect.fit_transform(X_train)
X_test_doc = vector.transform(X_test)
nb.fit(X_train_doc, y_train)
y_pred_text = nb.predict(X_test_doc)
print("Accuracy:")

print(metrics.accuracy_score(y_test, y_pred_text))

print()

print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, y_pred_text))
logreg = LogisticRegression()
logreg.fit(X_train_doc, y_train)
y_pred_text = logreg.predict(X_test_doc)
print("Accuracy:")

print(metrics.accuracy_score(y_test, y_pred_text))

print()

print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, y_pred_text))
file_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
Vector_test = vector.transform(test_df['text'])
y_predict_test = logreg.predict(Vector_test)

file_submission.target = y_predict_test

file_submission.to_csv("submission.csv", index=False)
file_submission.head(10)