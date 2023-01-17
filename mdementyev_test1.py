import pandas as pd

import seaborn as sns

import numpy as np

from collections import Counter



from sklearn.metrics import log_loss

from subprocess import check_output

from sklearn.cross_validation import train_test_split



import xgboost as xgb

import os

import gc



from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
p = df_train['is_duplicate'].mean()

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))
dist_train = train_qs.apply(len)

dist_test = test_qs.apply(len)



dist_train = train_qs.apply(lambda x: len(x.split(' ')))

dist_test = test_qs.apply(lambda x: len(x.split(' ')))
qmarks = np.mean(train_qs.apply(lambda x: '?' in x))

math = np.mean(train_qs.apply(lambda x: '[math]' in x))

fullstop = np.mean(train_qs.apply(lambda x: '.' in x))

capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))

capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))

numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))
def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
def get_weight(count, eps=100000000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)
eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
x_train = pd.DataFrame()

x_test = pd.DataFrame()



x_train['word_match'] = train_word_match

x_train['tfidf_word_match'] = tfidf_train_word_match

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]
p = 0.155

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=1234)

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.019

params['max_depth'] = 4
d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 735, watchlist, early_stopping_rounds=30, verbose_eval=2)

d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)
sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)