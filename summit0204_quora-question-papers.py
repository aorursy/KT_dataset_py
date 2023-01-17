# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv").fillna("")
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape)
print(df_test.shape)
df_train.info()
df_train['is_duplicate'].value_counts()
df_train.groupby("is_duplicate")['id'].count().plot.bar()
df_train['q1len']=df_train['question1'].str.len()
df_train['q2len']=df_train['question2'].str.len()
df_train['q1_n_words'] = df_train['question1'].apply(lambda row: len(row.split(" ")))
df_train['q2_n_words'] = df_train['question2'].apply(lambda row: len(row.split(" ")))
df_train.info()
def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

df_train['word_share'] = df_train.apply(normalized_word_share, axis=1)
df_train.head()
df_train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

# define tfidf vectorizer 
tfidf = TfidfVectorizer(analyzer = 'word',
                        stop_words = 'english',
                        lowercase = True,
                        max_features = 300,
                        norm = 'l1')
BagOfWords = pd.concat([df_train.question1, df_train.question2], axis = 0)
tfidf.fit(BagOfWords)
df_train['q1_tfidf'] = tfidf.transform(df_train.question1)
df_train['q2_tfidf'] = tfidf.transform(df_train.question2)
df_train['q1_tfidf'].shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(df_train[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])

X = scaler.transform(df_train[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])
y = df_train['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1', 'l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
cv.fit(X_train, y_train)
print(cv.best_params_)
print(cv.best_estimator_.coef_)
from sklearn.metrics import confusion_matrix
retrained = cv.best_estimator_.fit(X, y)
y_pred = retrained.predict(X)
confusion_matrix = confusion_matrix(df_train['is_duplicate'],y_pred)
print(confusion_matrix)
retrained.score(X,y)
ques = pd.concat([df_train[['question1', 'question2']], \
        df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
ques.shape
ques.head()
from collections import defaultdict
q_dict= defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
def q1_freq(row):
    return(len(q_dict[row['question1']]))
    
def q2_freq(row):
    return(len(q_dict[row['question2']]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
df_train['q2_freq'] =df_train.apply(q2_freq, axis=1, raw=True)
df_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(df_train[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share','q1_q2_intersect','q1_freq','q2_freq']])

X = scaler.transform(df_train[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share','q1_q2_intersect','q1_freq','q2_freq']])
y = df_train['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1', 'l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
cv.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
retrained = cv.best_estimator_.fit(X, y)
y_pred = retrained.predict(X)
confusion_matrix = confusion_matrix(df_train['is_duplicate'],y_pred)
print(confusion_matrix)
retrained.score(X,y)
