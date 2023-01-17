# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import string

import re

import nltk

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

import gc

import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.decomposition import TruncatedSVD

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/peepeepoopoo/training_data_sentiment.csv')

train.head()
test = pd.read_csv('../input/peepeepoopoo/testing_data_sentiment.csv')

test.head()
print('Size of train_data', train.shape)

print('Size of test_data', test.shape)
train['is_unsatisfied'].value_counts()
train['is_unsatisfied'].hist()
i = 3

q = train['question'][i].split('|||')

q = [i for i in q if i!='']

a = train['answer'][i].split('|||')

a = [i for i in a if i!='']



for i in range(max(len(a), len(q))):

    

    if i < len(q): print(q[i])

    else: print('No question')

    if i < len(a): print(a[i])

    else: print('No answer')

    print('---')
train_question = train['question'].str.len()

test_question = test['question'].str.len()



fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,6))

sns.distplot(train_question,ax=ax1,color='blue')

sns.distplot(test_question,ax=ax2,color='green')

ax2.set_title('Distribution for Question in test data')

ax1.set_title('Distribution for Question in Training data')



plt.show()
train_answer=train['answer'].str.len()

test_answer=test['answer'].str.len()



fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,6))

sns.distplot(train_answer,ax=ax1,color='blue')

sns.distplot(test_answer,ax=ax2,color='green')

ax2.set_title('Distribution for Answer in test data')

ax1.set_title('Distribution for Answer in Training data')



plt.show()
print("Number of duplicate questions in descending order")

print("------------------------------------------------------")

train.groupby('question').count()['session_id'].sort_values(ascending=False).head(25)
print("Number of duplicate answer in descending order")

print("------------------------------------------------------")

train.groupby('answer').count()['session_id'].sort_values(ascending=False).head(10)
train['is_train'] = 1

test['is_train'] = 0



df = train.append(test)

df
def clean_line(lines):

    

    lines = lines.split('|||')

    tmp = []

    for line in lines:

        line = line.split()

        line = [word.lower() for word in line]

        line = [word for word in line if word.isalpha()]

        line = ' '.join(line)

        tmp.append(line)

    return '|||'.join(tmp)
df['q_clean'] = [clean_line(line) for line in df['question']]

df['a_clean'] = [clean_line(line) for line in df['answer']]

df['q_count'] = [len(q.split('|||')) for q in df['q_clean']]

df['a_count'] = [len(a.split('|||')) for a in df['a_clean']]

df['q_all'] = [line.replace('|||', ' ') for line in df['q_clean']]

df['a_all'] = [line.replace('|||', ' ') for line in df['a_clean']]

df["q_num_chars"] = df["q_all"].apply(lambda x: len(str(x)))

df["a_num_chars"] = df["a_all"].apply(lambda x: len(str(x)))

df["q_num_words"] = df["q_clean"].apply(lambda x: len(str(x).split()))

df["a_num_words"] = df["a_clean"].apply(lambda x: len(str(x).split()))

df["q_unique"] = df["q_all"].apply(lambda x: len(set(str(x).split())))

df["a_unique"] = df["a_all"].apply(lambda x: len(set(str(x).split())))
feats = ['q_count', 'a_count', 'q_num_chars', 'a_num_chars', 'q_num_words', 'a_num_words', 'q_unique', 'a_unique']
cols = ['q', 'a']

n_tfidf = 500

for idx, col in enumerate(cols):

    print(f'TF-IDF {col}...')

    tfidf = TfidfVectorizer(analyzer='word',

                            token_pattern=r'\w{1,}',

                            ngram_range=(1, 3),

                            max_features= n_tfidf,

                            norm='l1',

                            sublinear_tf=True

                           )

    

    df[[col + '_' + str(i) for i in range(n_tfidf)]] = tfidf.fit_transform(df[col + '_all']).toarray()

    feats += [col + '_' + str(i) for i in range(n_tfidf)]

print('DONE!')
"""tfidf = TfidfVectorizer()

tsvd = TruncatedSVD(n_components = 128, n_iter=5)"""
"""q = tfidf.fit_transform(df["q_all"].values)

q_rep = tsvd.fit_transform(q)

a = tfidf.fit_transform(df["a_all"].values)

a_rep = tsvd.fit_transform(a)"""
"""df["q_rep"] = list(q_rep)

df['a_rep'] = list(a_rep)"""
train = df[df.is_train == 1]

test = df[df.is_train == 0]

test.drop(['is_unsatisfied'], inplace=True, axis=1)
print(train.shape, test.shape)
x = train[feats].values

y = train['is_unsatisfied'].apply(lambda x: 1 if x=='Y' else 0).values
hyper_params = {

    'task': 'train',

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'learning_rate': 0.01,

    "num_leaves": 250,

    "max_depth": 40,

    'feature_fraction': 0.85,

    "max_bin": 512,

    'metric': ['auc'],

    

    'subsample': 0.8,

    'subsample_freq': 2,

    'verbose': 0,

}

N_FOLD = 5



skf = StratifiedKFold(n_splits = N_FOLD, shuffle=True, random_state=13)



y_pred = np.zeros([y.shape[0]], dtype=float)

pred_sub = np.zeros([test.shape[0]], dtype=float)



for idx, (idx_train, idx_valid) in enumerate(skf.split(x, y)):

    print('Fold', idx, '...')

    

    gbm = lgb.LGBMModel(**hyper_params)

    gbm.fit(x[idx_train], y[idx_train],

            eval_set=[(x[idx_valid], y[idx_valid])],

            eval_metric='auc',

            verbose = 100,

            early_stopping_rounds=300)

    

    train_pred = gbm.predict(x[idx_train])

    valid_pred = gbm.predict(x[idx_valid])

    print('\tTrain:', metrics.accuracy_score(y[idx_train], train_pred.round()), metrics.f1_score(y[idx_train], train_pred.round()))

    print('\tValid:', metrics.accuracy_score(y[idx_valid], valid_pred.round()), metrics.f1_score(y[idx_valid], valid_pred.round()))

    

    y_pred[idx_valid] = valid_pred

    pred_sub += gbm.predict(test[feats].values) / N_FOLD
plt.figure(figsize=(10,10))

plt.plot(y,y)



# tmp = gbm.predict(X)

# fpr, tpr, thresholds = metrics.roc_curve(y, tmp, pos_label=1)

# plt.plot(fpr, tpr)



fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)

plt.plot(fpr, tpr)



metrics.roc_auc_score(y, y_pred)
test['num'] = test.index+1

test['label'] = pred_sub.round()

print(sum(test['label']))

test['label'] = test['label'].apply(lambda x: 'N' if x==0 else 'Y')



sub = test[['num','label']]



sub.to_csv('submission_lgbm.csv', index=False)



sub.label.value_counts()