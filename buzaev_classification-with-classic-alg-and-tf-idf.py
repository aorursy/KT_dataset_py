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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer
columns_name = ["target", "ids", "date", "flag", "user", "text"]

df = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv", encoding = 'ISO-8859-1', names = columns_name)
df.head()
df.info()
df['flag'].value_counts()
del df['flag']

del df['ids']

del df['date']
df.head()
df['target'].value_counts()
df['target'] = df['target'].map({0:0, 4:1})
print(df["target"].value_counts())

sns.barplot(df["target"].value_counts().index, df["target"].value_counts().values)

print('0 = negative, 1 = positive')
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 

          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',

          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 

          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',

          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',

          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 

          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
import string

# realization preprocessing

def preprocess(doc):

    # lower the text

    doc = doc.lower()

    # remove punctuation, spaces, etc.

    for p in string.punctuation + string.whitespace:

        doc = doc.replace(p, ' ')

    # remove extra spaces, merge back

    doc = doc.strip()

    doc = ' '.join([w for w in doc.split(' ') if w != ''])

    for emoji in emojis.keys():

        doc = doc.replace(emoji, "EMOJI" + emojis[emoji])

    return doc
for colname in df.select_dtypes(include = np.object).columns:

    df[colname] = df[colname].map(preprocess)

df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
negative_df = df[df['target'] == 0][:150000]

negative_df
positive_df = df[df['target'] == 1][:150000]

positive_df
df_limited = pd.DataFrame
df_limited = negative_df.append(positive_df, ignore_index=True)

df_limited = df_limited.sample(frac=1).reset_index(drop=True)

df_limited.head()
df_limited['target'].value_counts()
y = df_limited['target'].map({True: 1, False: 0}).values

y
df_limited.drop(['target'], axis = 1, inplace=True)

df_limited.head()
X = df_limited
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
vectorizer = TfidfVectorizer(stop_words = ENGLISH_STOP_WORDS, ngram_range=(1, 2)).fit(df['text'])



X_train_vectors = vectorizer.transform(X_train['text'])

X_test_vectors = vectorizer.transform(X_test['text'])
X_train_vectors.shape, X_test_vectors.shape
num = 65

X_train_vectors[num].data
vectorizer.inverse_transform(X_train_vectors[num])[0][np.argsort(X_train_vectors[num].data)]
knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train_vectors, y_train)
predicts = knn.predict((X_test_vectors))

print(classification_report(y_test, predicts))
lr = LogisticRegression(penalty = 'l2', C = 2, max_iter = 1000, n_jobs=-1).fit(X_train_vectors, y_train)
predicts = lr.predict((X_test_vectors))

print(classification_report(y_test, predicts))
clf = MultinomialNB(alpha = 2.0689655172413794).fit(X_train_vectors, y_train)
predicts = clf.predict((X_test_vectors))

print(classification_report(y_test, predicts))
clf = BernoulliNB(alpha = 2.0689655172413794).fit(X_train_vectors, y_train)
predicts = clf.predict((X_test_vectors))

print(classification_report(y_test, predicts))
clf = ComplementNB(alpha = 2.0689655172413794).fit(X_train_vectors, y_train)
predicts = clf.predict((X_test_vectors))

print(classification_report(y_test, predicts))