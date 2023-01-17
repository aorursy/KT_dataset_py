# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import gc
## read data from path



nrows = 10000000

# nrows = 10000



path = '../input/quora-question-pairs/'

train = pd.read_csv(path+"train.csv",nrows=nrows).astype(str)

test = pd.read_csv(path+"test.csv",nrows=nrows).astype(str)
train.head(10)
## eval metric is logloss

from sklearn.metrics import log_loss
## baseline with simple raw features

def try_divide(x, y, val=0.0):

    """ 

        Try to divide two numbers

    """

    if y != 0.0:

        val = float(x) / y

    return val



def get_jaccard(seq1, seq2):

    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.

    They should contain hashable items.

    

    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.

    """

    set1, set2 = set(seq1), set(seq2)

    return 1 -  try_divide(len(set1 & set2),float(len(set1 | set2)))



def get_dice(A,B):

    A, B = set(A), set(B)

    intersect = len(A.intersection(B))

    union = float(len(A) + len(B))

    d = try_divide(2*intersect, union)

    return d



def get_sorensen(seq1, seq2):

    """Compute the Sorensen distance between the two sequences `seq1` and `seq2`.

    They should contain hashable items.

    

    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.

    """

    set1, set2 = set(seq1), set(seq2)

    return 1-  try_divide(2 * len(set1 & set2),float(len(set1) + len(set2)))



def get_count_q1_in_q2(seq1,seq2):

    set1, set2 = set(seq1), set(seq2)

    return len(set1 & set2)



def get_ratio_q1_in_q2(seq1,seq2):

    set1, set2 = set(seq1), set(seq2)

    try:

        return len(set1 & set2)/float(len(set1))

    except:

        return 0.0



def get_count_of_question(seq1):

    return len(seq1)



def get_count_of_unique_question(seq1):

    set1 = set(seq1)

    return len(set1)



def get_ratio_of_unique_question(seq1):

    set1 = set(seq1)

    try:

        return len(set1)/float(len(seq1))

    except:

        return 0.0



def get_count_of_digit(seq1):

    return sum([1. for w in seq1 if w.isdigit()])



def get_ratio_of_digit(seq1):

    try:

        return sum([1. for w in seq1 if w.isdigit()])/float(len(seq1))

    except:

        return 0.0



def get_sim_feature(X_batch_1,X_batch_2):



    X_jaccard = np.array([ get_jaccard(x1,x2) for x1,x2 in zip(X_batch_1,X_batch_2)]).reshape(-1,1)

    X_dice = np.array([ get_dice(x1,x2)  for x1,x2 in zip(X_batch_1,X_batch_2)]).reshape(-1,1)

    X_count = np.array([ get_count_q1_in_q2(x1,x2)  for x1,x2 in zip(X_batch_1,X_batch_2)]).reshape(-1,1)

    X_ratio = np.array([ get_ratio_q1_in_q2(x1,x2)  for x1,x2 in zip(X_batch_1,X_batch_2)]).reshape(-1,1)

    X_len1 = np.array([ get_count_of_question(x1)  for x1 in  X_batch_1]).reshape(-1,1)

    X_len2 = np.array([ get_count_of_question(x2)  for x2 in  X_batch_2]).reshape(-1,1)



    X_len1_unique = np.array([ get_count_of_unique_question(x1)  for x1 in  X_batch_1]).reshape(-1,1)

    X_len2_unique = np.array([ get_count_of_unique_question(x2)  for x2 in  X_batch_2]).reshape(-1,1)



    X_len_diff = np.abs(X_len2-X_len1)





    X_batch_sim = np.hstack([X_jaccard,X_dice,X_count,X_ratio,X_len1,X_len2,X_len1_unique,X_len2_unique,X_len_diff])

    



    return X_batch_sim

# generate statistics features (sim and count of words)

from sklearn.preprocessing import StandardScaler,MinMaxScaler



X = get_sim_feature(train['question1'].apply(lambda x:x.split(' ')),train['question2'].apply(lambda x:x.split(' ')))

X_test = get_sim_feature(test['question1'].apply(lambda x:x.split(' ')),test['question2'].apply(lambda x:x.split(' ')))

y = train['is_duplicate'].values









scaler = StandardScaler()

X = scaler.fit_transform(X)

X_test = scaler.transform(X_test)



print(X.shape,X_test.shape)

# generate raw tfidf features sparse

from scipy import sparse as ssp

from sklearn.feature_extraction.text import TfidfVectorizer



corpus = []



feats = ['question1','question2']

for f in feats:

    train[f] = train[f].astype(str)

    test[f] = test[f].astype(str)

    corpus+=train[f].values.tolist()

    

ngram_range = (1,2)

max_features = 50000

min_df = 3

tfidf = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

tfidf.fit(corpus)



X_tfidf_q1 = tfidf.transform(train['question1'])

X_tfidf_q2 = tfidf.transform(train['question2'])

X_tfidf_test_q1 = tfidf.transform(test['question1'])

X_tfidf_test_q2 = tfidf.transform(test['question2'])



X_tfidf = ssp.hstack([X_tfidf_q1,X_tfidf_q2]).tocsr()

X_tfidf_test = ssp.hstack([X_tfidf_test_q1,X_tfidf_test_q2]).tocsr()



del X_tfidf_q1

del X_tfidf_q2

del X_tfidf_test_q1

del X_tfidf_test_q2

gc.collect()
## depomposition data

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(100)

svd.fit(ssp.vstack([X_tfidf,X_tfidf_test]))

X_svd = svd.transform(X_tfidf)

X_test_svd = svd.transform(X_tfidf_test)
## merge data

X = np.hstack([X,X_svd])

X_test = np.hstack([X_test,X_test_svd])

## validation set



from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,train_size=0.8,random_state=1024)

print(X_train.shape,X_val.shape)
## train lightgbm as classifier

import lightgbm as lgb



lgb_params = {"boosting": "gbdt", 'learning_rate': 0.05,

 "feature_fraction": 0.6, "bagging_freq": 1, "bagging_fraction": 0.8 , 'n_estimators': 100000,

 "metric": 'mae', "lambda_l1": 0.1, 'num_leaves': 32, 'min_data_in_leaf': 50, "verbose": 1, "num_threads": 8,

 "bagging_seed" : 1024,

 "seed": 1024,

 'feature_fraction_seed': 1024,

 }



clf = lgb.LGBMClassifier(**lgb_params)

clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], verbose = 100, eval_metric ='logloss',early_stopping_rounds=200)     



y_pred = clf.predict_proba(X_val)[:,1]

eval_score = log_loss(y_val,y_pred)

print(eval_score)



## make submission 

y_pred_test = clf.predict_proba(X_test)[:,1]



submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['is_duplicate'] = y_pred_test

submission.to_csv('submission_lgb.csv',index=False)
