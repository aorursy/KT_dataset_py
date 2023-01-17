# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns



from nltk.corpus import stopwords

from collections import Counter

%matplotlib inline
df_train = pd.read_csv( "../input/train.csv")

df_test = pd.read_csv( "../input/test.csv")

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()

                    +df_test['question1'].tolist() + df_test['question2'].tolist()

                    ).astype(str)
stops = set(stopwords.words("english"))



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

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

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

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    total_weights1 = [1 if x == 0 else x for x in total_weights]

            

    R = np.sum(shared_weights) / np.sum(total_weights1)

    return R
x_train = pd.DataFrame()
x_test = pd.DataFrame()
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

#train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
#tfidf_train_word_match1 = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

#x_test['tfidf_word_match'] = tfidf_train_word_match1

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

#x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
df_train = df_train.fillna("")

df_test =  df_test.fillna("")

x_train['q1len'] = df_train['question1'].str.len()

x_train['q2len'] = df_train['question2'].str.len()

x_train['q1_n_words'] = df_train['question1'].apply(lambda row: len(row.split(" ")))

x_train['q2_n_words'] = df_train['question2'].apply(lambda row: len(row.split(" ")))
x_test['q1len'] = df_test['question1'].str.len()

x_test['q2len'] = df_test['question2'].str.len()

x_test['q1_n_words'] = df_test['question1'].apply(lambda row: len(row.split(" ")))

x_test['q2_n_words'] = df_test['question2'].apply(lambda row: len(row.split(" ")))
def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
x_train['word_share'] = df_train.apply(normalized_word_share, axis=1)
x_test['word_share'] = df_test.apply(normalized_word_share, axis=1)
# First we create our training and testing data

x_train['word_match'] = train_word_match

x_train['tfidf_word_match'] = tfidf_train_word_match

y_train = df_train['is_duplicate'].values
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, 

                                                      test_size=0.2, random_state=42)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 40



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 180, watchlist, early_stopping_rounds=30, verbose_eval=10)
#x_test.columns = ['word_match', 'q1len','q1_n_words','word_share','tfidf_word_match']

x_test1 = x_test[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share',

       'word_match', 'tfidf_word_match']]

#x_test
d_test = xgb.DMatrix(x_test1)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test
sub.to_csv('multi_feature_xgb.csv', index=False)