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
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.isnull().sum()
test.isnull().sum()
test.info
train.drop(columns = 'location', inplace = True)

test.drop(columns = 'location', inplace = True)
train[train['keyword'].notnull()][train['target']== 1]
train[train['keyword'].notnull()][train['target']== 0]
train['keyword'].value_counts().index
train.head(10)
def lowercase_text(text):

    text = text.lower()

    return text

train['text'] = train['text'].apply(lambda x : lowercase_text(x))

test['text'] = test['text'].apply(lambda x : lowercase_text(x))

train['text'].head(10)
import string

string.punctuation
train.head(10)
def remove_punctuation(text):

    text_no_punctuation = "".join([c for c in text if c not in string.punctuation])

    return text_no_punctuation

train["text"] = train["text"].apply(lambda x: remove_punctuation(x))

test["text"] = test["text"].apply(lambda x: remove_punctuation(x))
train['text'].head(10)
import nltk

from nltk.tokenize import RegexpTokenizer

# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
from nltk.corpus import stopwords

print(stopwords.words('english'))
train.head(10)
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head(10)
def combine_text(list_of_text):

    combine_text = ' '.join(list_of_text)

    return combine_text

train["text"] = train["text"].apply(lambda x: combine_text(x))

test["text"] = test["text"].apply(lambda x: combine_text(x))
train.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 



train_vector = count_vectorizer.fit_transform(train["text"]).todense()

test_vector = count_vectorizer.transform(test["text"]).todense()
print(count_vectorizer.vocabulary_)
print(train_vector.shape)

print(test_vector.shape)
from sklearn.model_selection import train_test_split



Y = train["target"]

x_train, x_test, y_train,y_test = train_test_split(train_vector,Y, test_size = 0.3, random_state = 0)

y_train
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 3.0)

scores = cross_val_score(model, train_vector, train['target'], cv=5)

print(scores.mean())
model.fit(x_train, y_train)

y_pred_model_1 = model.predict(x_test)

print(accuracy_score(y_test,y_pred_model_1))
y_pred_test = model.predict(test_vector)
from sklearn.svm import SVC
svm = SVC(kernel = "linear", C = 0.15, random_state = 100)

svm.fit(x_train, y_train)

y_pred = svm.predict(test_vector)
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv("submission.csv", index=False)

sub.head(10)