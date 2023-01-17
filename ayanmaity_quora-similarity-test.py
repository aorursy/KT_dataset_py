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
train_df = pd.read_csv('../input/train.csv')

train_df.head()
train_df.tail()
train_df['text'] = train_df['question1']+ " " + train_df['question2']
train_df['text'].values.shape
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import text

st_wd = text.ENGLISH_STOP_WORDS

tf_vec = CountVectorizer(stop_words=st_wd,min_df=0.00001,max_df=.7)

tf_vec.fit(train_df['text'].values.astype('U'))
len(tf_vec.vocabulary_)
q1_tf = tf_vec.transform(train_df['question1'].values.astype('U'))

q2_tf = tf_vec.transform(train_df['question2'].values.astype('U'))
q1_tf_len = np.sum(q1_tf.multiply(q1_tf),axis=1)

q2_tf_len = np.sum(q2_tf.multiply(q2_tf),axis=1)
np.sum(q1_tf_len)
q1_tf_len = np.sqrt(q1_tf_len)

q2_tf_len = np.sqrt(q2_tf_len)
np.sum(q1_tf_len)
q1_tf_len = np.array(q1_tf_len)

q2_tf_len = np.array(q2_tf_len)
prod = q1_tf_len*q2_tf_len
mul_vec = q1_tf.multiply(q2_tf)
mul_vec 
feature = np.sum(mul_vec,axis=1)

feature.shape
feature = feature/prod 
feature = np.array(feature)
feature.shape
feature[np.isfinite(feature)==False]=.0001
np.sum(np.isfinite(feature))
feature[feature==.0001]
diff_vec = q1_tf-q2_tf
diff_vec
diff_vec = diff_vec.multiply(diff_vec)
diff_vec = np.sum(diff_vec,axis=1)
diff_vec = np.array(diff_vec)
diff_vec = np.sqrt(diff_vec)
diff_vec
diff_vec = diff_vec/np.max(diff_vec)
diff_vec
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='sgd',hidden_layer_sizes = (70),max_iter=250,verbose=1,learning_rate_init=.08,random_state=1)
X_train = np.array(feature)/np.max(feature)

X_train = np.append(X_train,diff_vec,axis=1)

q1_sum = q1_tf.sum(axis=1)

q2_sum = q2_tf.sum(axis=1)

X_train = np.append(X_train,q1_sum,axis=1)

X_train = np.append(X_train,q2_sum,axis=1)

X_train = np.append(X_train,q1_tf_len,axis=1)

X_train = np.append(X_train,q2_tf_len,axis=1)

X_train
target = train_df['is_duplicate'].values
clf.fit(X_train,target)
clf.score(X_train,target)
import xgboost as xgb

train_Y = target

xg_train = xgb.DMatrix(X_train, label=train_Y)

##xg_test = xgb.DMatrix(test_X, label=test_Y)

# setup parameters for xgboost

param = {}

# use softmax multi-class classification

param['objective'] = 'multi:softprob'

# scale weight of positive examples

param['eta'] = .8

param['max_depth'] = 10

param['silent'] = 1

param['nthread'] = 5

param['num_class'] = 2

param['eval_metric'] = "mlogloss"



watchlist = [(xg_train, 'train')]

num_round = 500

bst = xgb.train(param, xg_train, num_round, watchlist)

# get prediction

pred = bst.predict(xg_train)

pred_Y = np.argmax(pred,axis=1)

print(pred_Y)

score = np.sum(pred_Y == train_Y) / X_train.shape[0]

print('Train score using softprob = {}'.format(score))