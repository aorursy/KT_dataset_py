import sys

sys.path
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

print("This is just a test")



quora_train = pd.read_csv("../input/train.csv")

quora_test = pd.read_csv("../input/test.csv")


#quora_train.info()

#quora_test.info()



#print('-'*40)

#print(quora_train.tail())

#print('-'*40)

#print(quora_test.tail())
print('Total number of question pairs for training: {}'.format(len(quora_train)))

print('Duplicate pairs: {}%'.format(round(quora_train['is_duplicate'].mean()*100, 2)))

qids = pd.Series(quora_train['qid1'].tolist() + quora_train['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(

    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
from sklearn.metrics import log_loss as lgl

p = quora_train['is_duplicate'].mean()

print('Predicted score:', lgl(quora_train['is_duplicate'], 

                              np.zeros_like(quora_train['is_duplicate']) + p))

sub = pd.DataFrame({'test_id': quora_test['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index = False)

print(sub.head())
from sklearn.metrics import log_loss as lgl

p = quora_train['is_duplicate'].mean()

print('Predicted score:', lgl(quora_train['is_duplicate'], 

                              np.zeros_like(quora_train['is_duplicate']) + p))

sub = pd.DataFrame({'test_id': quora_test['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index = False)

print(sub.head())
from nltk.corpus import stopwords



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

print(stops)



plt.figure(figsize=(15, 5))

train_word_match = quora_train.apply(word_match_share, axis=1, raw=True)

plt.hist(train_word_match[quora_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match[quora_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
train_qs = pd.Series(quora_train['question1'].tolist() + quora_train['question2'].tolist()).astype(str)

test_qs = pd.Series(quora_test['question1'].tolist() + quora_test['question2'].tolist()).astype(str)



dist_train = train_qs.apply(len)

dist_test = test_qs.apply(len)

print('ok')
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

train_qs.head(5)

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

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R


tfidf_train_word_match = quora_train.apply(tfidf_word_match_share, axis=1, raw=True)

plt.hist(tfidf_train_word_match[quora_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')

plt.hist(tfidf_train_word_match[quora_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
# First we create our training and testing data

x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train['word_match'] = train_word_match

x_train['tfidf_word_match'] = tfidf_train_word_match

x_test['word_match'] = quora_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = quora_test.apply(tfidf_word_match_share, axis=1, raw=True)



x_train['gen2'] = np.sqrt(train_word_match)

x_train['gen2_tfidf'] = np.sqrt(tfidf_train_word_match)

x_test['gen2'] = np.sqrt(x_test['tfidf_word_match'])

x_test['gen2_tfidf'] = np.sqrt(x_test['tfidf_word_match'])



x_train['sin_w'] = np.sin(train_word_match)

x_train['sin_tf'] = np.sin(tfidf_train_word_match)

x_test['sin_w'] = np.sin(x_test['word_match'])

x_test['sin_tf'] = np.sin(x_test['tfidf_word_match'])



x_train['2w'] = train_word_match * train_word_match

x_train['2tf'] = tfidf_train_word_match * tfidf_train_word_match

x_test['2w'] = x_test['word_match'] * x_test['word_match']

x_test['2tf'] = x_test['tfidf_word_match'] * x_test['tfidf_word_match']



y_train = quora_train['is_duplicate'].values



print('OK')
import timeit



train_orig =  pd.read_csv('../input/train.csv', header=0)

test_orig =  pd.read_csv('../input/test.csv', header=0)





tic0=timeit.default_timer()

df1 = train_orig[['question1']].copy()

df2 = train_orig[['question2']].copy()

df1_test = test_orig[['question1']].copy()

df2_test = test_orig[['question2']].copy()



df2.rename(columns = {'question2':'question1'},inplace=True)

df2_test.rename(columns = {'question2':'question1'},inplace=True)



train_questions = df1.append(df2)

train_questions = train_questions.append(df1_test)

train_questions = train_questions.append(df2_test)

train_questions.info()



train_questions.drop_duplicates(subset = ['question1'],inplace=True)



train_questions.reset_index(inplace=True,drop=True)

questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()

train_cp = train_orig.copy()

test_cp = test_orig.copy()

train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1

test_cp.rename(columns={'test_id':'id'},inplace=True)

comb = pd.concat([train_cp,test_cp])



comb['q1_hash'] = comb['question1'].map(questions_dict)

comb['q2_hash'] = comb['question2'].map(questions_dict)



print('ok')
q1_vc = comb.q1_hash.value_counts().to_dict()

q2_vc = comb.q2_hash.value_counts().to_dict()



def try_apply_dict(x,dict_to_apply):

    try:

        return dict_to_apply[x]

    except KeyError:

        return 0

# map to frequency space

comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

print('test')

train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]

test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

train_comb.drop(['is_duplicate'],axis=1,inplace=True)

print('ok')


x_train = pd.concat([x_train, train_comb], axis=1)

x_test = pd.concat([x_test, test_comb], axis=1)

x_train = pd.concat([x_train, train_comb], axis=1)

x_test = pd.concat([x_test, test_comb], axis=1)

#x_train = x_train.drop(['q1_hash','q2_hash'], axis = 1)

#x_test = x_test.drop(['q1_hash','q2_hash'], axis = 1)

x_train.info()

x_test.info()

print('ok')
# may be del.  I will try.

#pos_train = x_train[y_train == 1]

#neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

#p = 0.165

#scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

#while scale > 1:

#    neg_train = pd.concat([neg_train, neg_train])

#    scale -=1

#neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

#print(len(pos_train) / (len(pos_train) + len(neg_train)))



#x_train = pd.concat([pos_train, neg_train])

#y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

#del pos_train, neg_train

print('ok')
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

print('ok')
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.05

params['max_depth'] = 4

params['min_child_weight'] = 3

params['gamma'] = 0

params['subsample'] = 0.9

params['colsample_bytree'] = 0.9

params['nthread'] = 4

params['seed'] = 50

params['scale_pos_weight=1'] = 1



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 450, watchlist, early_stopping_rounds=50, verbose_eval=10)

print('ok')
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = quora_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)

print('ok')