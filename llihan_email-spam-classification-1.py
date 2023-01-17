# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pylab as pl

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

from subprocess import check_output

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report,confusion_matrix

from wordcloud import WordCloud

# Any results you write to the current directory are saved as output.

import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

import string

from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD

import pandas as pd
messages = pd.read_csv('../input/spam.csv', encoding='latin-1')
messages.head()
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
messages = messages.rename(columns={'v1': 'class', 'v2':'text'})
messages.head()
messages['words_len'] = messages['text'].apply(lambda x: len(x.split(' ')))

messages['char_len'] = messages['text'].apply(len)
f, ax = plt.subplots(figsize=(12, 8))

sns.stripplot(x="class", y="words_len", data=messages)

plt.title('Number of words per class')
f, ax = plt.subplots(figsize=(12, 8))

sns.stripplot(x="class", y="char_len", data=messages)

plt.title('Number of characters per class')
def process_text(text):

    # 1. Remove punctuation

    # 2. Remove stop words

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    clean_words = [w for w in nopunc.split() if w.lower() not in stopwords.words('english')]

    return clean_words



messages['text'].apply(process_text).head()
X_train, X_test, y_train, y_test = train_test_split(messages['text'], messages['class'], test_size=0.2)
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=process_text)),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB())

])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
import seaborn as sns

sns.heatmap(confusion_matrix(y_test, predictions),annot=True)
from sklearn.feature_extraction.text import TfidfVectorizer

vectors = TfidfVectorizer().fit_transform(messages.text)
X_reduced = TruncatedSVD(n_components=100, random_state=0).fit_transform(vectors)

tsne = TSNE(n_components=2, perplexity=110, verbose=2).fit_transform(X_reduced)
import hypertools as hyp

hyp.plot(tsne,'o', group=messages['class'], legend=list({'ham':0, 'spam':1}))