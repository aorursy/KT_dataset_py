import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
input = '../input/'
df_train = pd.read_csv(input+"train.csv")

df_test = pd.read_csv(input+"test.csv")
qids = pd.Series(df_train['qid1'].tolist()+df_train['qid2'].tolist())
train_qs = pd.Series(df_train["question1"].tolist()+df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test["question1"].tolist()+df_test['question2'].tolist()).astype(str)
import re

r = re.compile("[ ,.?]")
from nltk.corpus import stopwords



stops = set(stopwords.words('english'))|set([''])
def word_match(row):

    q1_word = {}

    q2_word = {}

    for word in r.split(str(row["question1"]).lower()):

        if word not in stops:

            q1_word[word] = 1

    for word in r.split(str(row["question2"]).lower()):

        if word not in stops:

            q2_word[word] = 1

    if len(q1_word) == 0 or len(q2_word)==0:

        return 0

    shared_word_in_q1 = [w for w in q1_word.keys() if w in q2_word]

    shared_word_in_q2 = [w for w in q2_word.keys() if w in q1_word]

    R = (len(shared_word_in_q1)+len(shared_word_in_q2))/(len(q1_word)+len(q2_word))

    return R



train_word_related = df_train.apply(word_match,axis = 1,raw = 1)
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = r.split((" ".join(train_qs)).lower())

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items() if word != ""}
print("Most commonly used words are: \n")

print(sorted(weights.items(),key=lambda x: x[1] if x[1]>0 else 9999)[:10])

print("\nLeast commonly used words are: \n")

print(sorted(weights.items(),key=lambda x: x[1],reverse = True)[:10])
def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in r.split(str(row['question1']).lower()):

        if word not in stops:

            q1words[word] = 1

    for word in r.split(str(row['question2']).lower()):

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
tfidf_train_word_share = df_train.apply(tfidf_word_match_share,axis = 1, raw = True)
from sklearn.metrics import roc_auc_score

print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_related))

print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_share.fillna(0)))
# First we create our training and testing data

x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train['word_match'] = train_word_related

x_train['tfidf_word_match'] = tfidf_train_word_share

x_test['word_match'] = df_test.apply(word_match, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
x_train
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)




d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test



sub

sub.to_csv('prediction2049.csv', index=False)

sub2 = pd.read_csv('prediction2049.csv')

sub2