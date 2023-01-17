import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold, train_test_split



sns.set()



%matplotlib inline
DATA_DIR = '/kaggle/input/texts-classification-iad-hse-intro-2020/'



df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
df_train.head()
df_test.head()
sub.head()
df_train.shape, df_test.shape
df_train['Category'].value_counts()
len(df_train['Category'].unique())
plt.figure(figsize=(11, 8))

df_train['Category'].hist(bins=50)

plt.show()
skf = StratifiedKFold(n_splits=10)



for train_index, test_index in skf.split(df_train, df_train['Category']):

    df_train_small = df_train.iloc[test_index]

    break



df_train_small.shape, df_train.shape
plt.figure(figsize=(11, 8))

df_train_small['Category'].hist(bins=50)

plt.show()
X_train, X_val, y_train, y_val = train_test_split(df_train_small, df_train_small['Category'], random_state=13)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
del df_train
X_train.head()
%%time



tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train['title'])
len(tfidf.vocabulary_)
X_train_tfidf.shape
%%time



lr = LogisticRegression(random_state=13)

lr.fit(X_train_tfidf, y_train)
%%time



X_val_tfidf = tfidf.transform(X_val['title'])
%%time



y_val_pred = lr.predict(X_val_tfidf)

accuracy_score(y_val, y_val_pred)
%%time



X_test_tfidf = tfidf.transform(df_test['title'])

y_test_pred = lr.predict(X_test_tfidf)
sub.shape, y_test_pred.shape
sub['Category'] = y_test_pred

sub.head()
plt.figure(figsize=(11, 8))

sub['Category'].hist(bins=50)

plt.show()
sub.to_csv('submission.csv', index=False)