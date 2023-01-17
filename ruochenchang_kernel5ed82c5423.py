import numpy as np 

import pandas as pd 



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train_data.head() #dataframe
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test_data.head() #dataframe
train_data.info()

test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.describe()
train_data['target'].value_counts()
def clean_text(text):

    # Make text lowercase, remove punctuation.

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    return text



# Applying the cleaning function to both test and training datasets

train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))

test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))



train_data['text'].head()
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train_data['text'] = train_data['text'].apply(lambda x: tokenizer.tokenize(x))

test_data['text'] = test_data['text'].apply(lambda x: tokenizer.tokenize(x))

train_data['text'].head()
def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words



train_data['text'] = train_data['text'].apply(lambda x : remove_stopwords(x))

test_data['text'] = test_data['text'].apply(lambda x : remove_stopwords(x))

train_data.head()
# After preprocessing, the text format

def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text



train_data['text'] = train_data['text'].apply(lambda x : combine_text(x))

test_data['text'] = test_data['text'].apply(lambda x : combine_text(x))

train_data['text']

train_data.head()
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_data['text'])

test_vectors = count_vectorizer.transform(test_data["text"])



print(train_vectors[0].todense())

print(test_vectors.shape[1])

# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train_data["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_data["target"])
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv('my_submission.csv', index=0)

print("success")