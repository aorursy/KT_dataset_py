# import necessary Python libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # operating system

import gc # garbage collecting

import matplotlib.pyplot as plt # plotting

import seaborn as sns # statistical data visualization

from __future__ import division # division support in Python 2.7

import nltk # Natural Language Processing

import codecs # decoding

import pickle # for saving off large files

import re

%matplotlib inline
# setting a color palette for plotting and importing the train and test files

pal = sns.color_palette()

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head(10)
df_test.head(10)
df_train_clean = df_train

for tag in ['question1', 'question2']:

    df_train_clean[tag] = df_train[tag].str.lower()

    df_train_clean[tag].replace(to_replace=r"[^A-Za-z0-9^,!.\/'+-=]", value=" ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"what's", value="what is ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"'s", value=" ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"'ve'", value=" have ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"can't", value="cannot ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"n't", value=" not ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"i'm", value="i am ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"'re'", value=" are ", regex=True, inplace=True)

    df_train_clean[tag].replace(to_replace=r"'ll'", value=" will ", regex=True, inplace=True)
# x_train and x_test will be used to store the different selected features

x_train = pd.DataFrame()

x_test = pd.DataFrame()
qdict = {'which': 'determiner', 

         'what': 'determiner',

         'whose': 'personal determiner',

         'who': 'personal determiner',

         'whom': 'personal determiner',

         'where': 'location',

         'whither': 'goal',

         'whence': 'source',

         'how': 'manner',

         'why': 'reason',

         'whatsoever': 'choice',

         'whether': 'choice'

        }
def question_match(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word in qdict.keys():

            q1words[qdict[word]] = 1

    for word in str(row['question2']).lower().split():

        if word in qdict.keys():

            q2words[qdict[word]] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared = [w for w in q1words.keys() if w in q2words]

    R = len(shared)*2/(len(q1words) + len(q2words))

    return R



train_q_match = df_train_clean.apply(question_match, axis=1, raw=True)

plt.figure(figsize=(15, 5))

plt.hist(train_q_match[df_train_clean['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_q_match[df_train_clean['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over question_match_share', fontsize=15)

plt.xlabel('question_match_share', fontsize=15)
x_train['q_match'] = train_q_match

x_test['q_match'] = df_test.apply(question_match, axis=1, raw=True)
from nltk.corpus import stopwords # stopwords are common words best ignored because the relay no meaning (e.g. "The")



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

    if len(q1words) == 0 or len(q2words) == np.nan:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/((len(q1words)  + len(q2words)))

    return R



plt.figure(figsize=(15, 5))

train_word_match = df_train_clean.apply(word_match_share, axis=1, raw=True)

plt.hist(train_word_match[df_train_clean['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match[df_train_clean['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
x_train['word_match'] = train_word_match

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
### Total number of words in both questions

def word_total_combined(row):

    q1words = {}

    q2words = {}

    R = 0

    for word in str(row['question1']).lower().split():

        R = R + 1

    for word in str(row['question2']).lower().split():

        R = R + 1

    return R



plt.figure(figsize=(15, 5))

train_word_total = df_train_clean.apply(word_total_combined, axis=1, raw=True)

plt.hist(train_word_total[df_train_clean['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_total[df_train_clean['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_total_combined', fontsize=15)

plt.xlabel('word_total_combined', fontsize=15)
x_train['word_total_combined'] = train_word_total

x_test['word_total_combined'] = df_test.apply(word_total_combined, axis=1, raw=True)
def word_diff(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():       

        q1words[word] = 1

    for word in str(row['question2']).lower().split():

        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    R = abs(len(q1words) - len(q2words))

    return R



plt.figure(figsize=(15, 5))

train_word_diff = df_train_clean.apply(word_diff, axis=1, raw=True)

plt.hist(train_word_diff[df_train_clean['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_diff[df_train_clean['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_diff', fontsize=15)

plt.xlabel('word_diff', fontsize=15)
x_train['word_diff'] = train_word_total

x_test['word_diff'] = df_test.apply(word_total_combined, axis=1, raw=True)
x_test.head()
from collections import Counter



train_qs = pd.Series(df_train_clean['question1'].tolist() + df_train_clean['question2'].tolist()).astype(str)



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
print('Most common words and weights: \n')

print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

print('\nLeast common words and weights: ')

(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



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

    

    missing_weights = np.sum([weights.get(w, 0) for w in q1words.keys() if w not in q2words] + [weights.get(w, 0) for w in q2words.keys() if w not in q1words])

    #total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    #R = np.sum(shared_weights) / np.sum(total_weights)

    return missing_weights
plt.figure(figsize=(15, 5))

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')

plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over tfidf_word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
x_train['tfidf'] = tfidf_train_word_match

x_test['tfidf'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
y_train = df_train['is_duplicate'].values
# 

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

0.19124366100096607
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2237)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 6

params['min_child_weight'] = 4





d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
import operator



def ceate_feature_map(features):

    outfile = open('xgb.fmap', 'w')

    i = 0

    for feat in features:

        outfile.write('{0}\t{1}\tq\n'.format(i, feat))

        i = i + 1



    outfile.close()



features = list(x_train.columns)

ceate_feature_map(features)



importance = bst.get_fscore(fmap='xgb.fmap')

importance = sorted(importance.items(), key=operator.itemgetter(1))



df = pd.DataFrame(importance, columns=['feature', 'fscore'])

df['fscore'] = df['fscore'] / df['fscore'].sum()



plt.figure()

df.plot()

df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

plt.title('XGBoost Feature Importance')

plt.xlabel('relative importance')
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('submission.csv', index=False)