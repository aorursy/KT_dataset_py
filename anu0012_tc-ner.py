import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/to_contestant/to_contestant/train.txt', sep=" |", header=None)

test = pd.read_csv('../input/to_contestant/to_contestant/test_raw.txt', sep=" |", header=None)

dev = pd.read_csv('../input/to_contestant/to_contestant/test_raw.txt', sep=" |", header=None, names=['text'])
train = train.rename(columns={0: 'text', 1: 'label'})

test = test.rename(columns={0: 'text'})

dev = dev.rename(columns={0: 'text'})
train.head()
train['label'].unique()
mapping = {'B-PER':'PER', 'I-PER':'PER', 'B-LOC':'LOC', 'I-LOC':'LOC', 'B-ORG':'ORG','I-ORG':'ORG',

          'B-MIS':'MIS','I-MIS':'MIS'}

train['label'].replace(mapping,inplace=True)
sns.countplot(train['label'])
from sklearn.utils import resample

# Separate majority and minority classes

df_majority = train[train.label=='O']

df_minority1 = train[train.label=='MIS']

df_minority2 = train[train.label=='LOC']

df_minority3 = train[train.label=='PER']

df_minority4 = train[train.label=='ORG']

 

# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=2500,     # to match minority class

                                 random_state=123) # reproducible results

 

# Combine minority class with downsampled majority class

df_downsampled = pd.concat([df_majority_downsampled, df_minority1, df_minority2, df_minority3, df_minority4])

 

# Display new class counts

df_downsampled.label.value_counts()
from sklearn.model_selection import train_test_split



train_df, test_df = train_test_split(df_downsampled, test_size = 0.2)

train_arr = []

test_arr = []

train_lbl = []

test_lbl = []
train_arr=train_df['text'].astype(str)

train_lbl=train_df['label'].astype(str)

test_arr=test_df['text'].astype(str)

test_lbl=test_df['label'].astype(str)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
vectorizer = CountVectorizer()

vectorizer.fit(train_arr)

train_mat = vectorizer.transform(train_arr)
tfidf = TfidfTransformer()

tfidf.fit(train_mat)

train_tfmat = tfidf.transform(train_mat)

test_mat = vectorizer.transform(test_arr)

test_tfmat = tfidf.transform(test_mat)

del test_arr

del train_arr
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

import xgboost as xgb
train_tfmat
from sklearn.svm import LinearSVC,SVC

from sklearn.linear_model import LogisticRegression

import xgboost as xgb
lsvm= SVC(kernel='linear', 

            class_weight='balanced', # penalize

            probability=True)

lsvm.fit(train_tfmat,train_lbl)
y_pred_lsvm=lsvm.predict(test_tfmat)
sample_test=['ألمانيا']

test_str = vectorizer.transform(sample_test)

test_tfstr = tfidf.transform(test_str)

test_tfstr.shape

lsvm.predict(test_tfstr.toarray())[0]
from sklearn.metrics import  accuracy_score

from sklearn import metrics
import matplotlib.pyplot as plt

sns.countplot(test_lbl)

plt.show()

sns.countplot(y_pred_lsvm)

plt.show()
print("accuracy:", metrics.accuracy_score(test_lbl, y_pred_lsvm))
import sys

y=[]

token=[]

scores = []

for x in list(test['text']):

    x=[x]

    test_str = vectorizer.transform(x)

    test_tfstr = tfidf.transform(test_str)

    test_tfstr.shape

    token.append(x)

    scores.append(lsvm.predict_proba(test_tfstr.toarray()).max())

    y.append(lsvm.predict(test_tfstr.toarray())[0])
start = []

end = []

type_list = []

score_list = []



for i in range(len(y)):

    if y[i] == 'O':

        continue

    type_list.append(y[i])

    score_list.append(scores[i])

    start.append(i+1)

    end.append(i+1)

    



filenames = ['test.txt'] * len(type_list)

surface = ['?'] * len(type_list)

#scores = [1.0] * len(type_list)
sub = pd.DataFrame()

sub['Filename'] = filenames

sub['Start'] = start

sub['End'] = end

sub['Type'] = type_list

sub['Score'] = score_list

sub['Surface'] = surface

sub.to_csv('solution.csv',index=False)
y=[]

token=[]

scores = []

for x in list(dev['text']):

    x=[x]

    test_str = vectorizer.transform(x)

    test_tfstr = tfidf.transform(test_str)

    test_tfstr.shape

    token.append(x)

    scores.append(lsvm.predict_proba(test_tfstr.toarray()).max())

    y.append(lsvm.predict(test_tfstr.toarray())[0])

    

start = []

end = []

type_list = []

score_list = []

for i in range(len(y)):

    if y[i] == 'O':

        continue

    type_list.append(y[i])

    score_list.append(scores[i])

    start.append(i+1)

    end.append(i+1)

    



filenames = ['dev.txt'] * len(type_list)

surface = ['?'] * len(type_list)

#scores = [1.0] * len(type_list)



sub = pd.DataFrame()

sub['Filename'] = filenames

sub['Start'] = start

sub['End'] = end

sub['Type'] = type_list

sub['Score'] = score_list

sub['Surface'] = surface

sub.to_csv('solution_dev.csv',index=False)
y=[]

token=[]

scores = []

for x in list(train['text']):

    x=[x]

    test_str = vectorizer.transform(x)

    test_tfstr = tfidf.transform(test_str)

    test_tfstr.shape

    token.append(x)

    scores.append(lsvm.predict_proba(test_tfstr.toarray()).max())

    y.append(lsvm.predict(test_tfstr.toarray())[0])

    

start = []

end = []

type_list = []

score_list = []

for i in range(len(y)):

    if y[i] == 'O':

        continue

    type_list.append(y[i])

    score_list.append(scores[i])

    start.append(i+1)

    end.append(i+1)

    



filenames = ['train.txt'] * len(type_list)

surface = ['?'] * len(type_list)

#scores = [1.0] * len(type_list)



sub = pd.DataFrame()

sub['Filename'] = filenames

sub['Start'] = start

sub['End'] = end

sub['Type'] = type_list

sub['Score'] = score_list

sub['Surface'] = surface

sub.to_csv('solution_train.csv',index=False)