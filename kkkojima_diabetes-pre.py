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
#再現性の確保

#乱数の固定

import os

import random as rn



os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(7)

rn.seed(7)
from sklearn.model_selection import cross_validate, KFold, cross_val_score, train_test_split, cross_val_predict

from sklearn.metrics import make_scorer,roc_auc_score

import xgboost as xgb
train_df = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv', index_col=0)
print(train_df.shape)

print(test_df.shape)
#目的変数の確認

print(list(train_df['Diabetes']).count(1))

print(list(train_df['Diabetes']).count(0))
#train_df['Gender'] = train_df['Gender'].map({'male':0, 'female':1})

#test_df['Gender'] = test_df['Gender'].map({'male':0, 'female':1})
train_df = pd.get_dummies(train_df, drop_first=True)

test_df = pd.get_dummies(test_df, drop_first=True)
#train_df['Chol/HDL ratio labels'] = 0

#test_df['Chol/HDL ratio labels'] = 0
#for i in range(train_df.shape[0]):

 #   if train_df['Chol/HDL ratio'].iloc[i] < 5.0:

  #      train_df['Chol/HDL ratio labels'].iloc[i] = 0

   # else:

    #    train_df['Chol/HDL ratio labels'].iloc[i] = 1
#for i in range(test_df.shape[0]):

 #   if test_df['Chol/HDL ratio'].iloc[i] < 5.0:

  #      test_df['Chol/HDL ratio labels'].iloc[i] = 0

   # else:

    #    test_df['Chol/HDL ratio labels'].iloc[i] = 1
train_df
X_train = train_df.drop('Diabetes', axis=1).values

y_train = train_df['Diabetes'].values

print(X_train.shape)

print(y_train.shape)
#X_learn, X_valid, y_learn, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

#print('学習用データ:', X_learn.shape, y_learn.shape)

#print('検証用データ:', y_valid.shape, y_valid.shape)
#from sklearn.model_selection import GridSearchCV

#import xgboost as xgb

#reg = xgb.XGBClassifier()

#params = {'random_state':[0, 1], 'n_estimators':[100, 300, 500, 1000], 'max_depth':[1, 2, 3, 4, 5, 6],

 #         'learning_rate':[0.5, 0.1, 0.05, 0.01]}

#gscv = GridSearchCV(reg, params, cv=3)

#gscv.fit(X_res, y_res)
#CrossValidation

from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV

from keras.utils.np_utils import to_categorical



kf = KFold(n_splits=5, shuffle=False)

score_l = []

score_v = []



for i in kf.split(X_train, y_train):

    X_learn_s, y_learn_s = X_train[i[0]], y_train[i[0]]

    X_valid_s, y_valid_s = X_train[i[1]], y_train[i[1]]



    #不均衡データ対策

    ros = SMOTE(random_state=0)

    X_res, y_res = ros.fit_sample(X_learn_s, y_learn_s)

    Y_learn_s = to_categorical(y_learn_s)

    Y_valid_s = to_categorical(y_valid_s)

    #オーバーサンプリング後に行う前処理

    #X_res = np.insert(X_res, 13, 0, axis=1)

    #a = np.insert(a, 2, 1, axis=1)

    #for n in range(X_res.shape[0]):

     #   if X_res[n][2] < 5.0:

      #      X_res[n][13] = 0

       # if X_res[n][2] >= 5.0:

        #    X_res[n][13] = 1

    

    clf = LGBMClassifier()

    clf.fit(X_res, y_res)

    

    pre_l = clf.predict_proba(X_learn_s)

    pre_v = clf.predict_proba(X_valid_s)

    

    score_l.append(roc_auc_score(Y_learn_s[:,1], pre_l[:,1]))

    score_v.append(roc_auc_score(Y_valid_s[:,1], pre_v[:,1]))





print('CrossValidation(learn_data) : ', sum(score_l)/len(score_l))

print('CrossValidation(val_data) : ', sum(score_v)/len(score_v))
X_res[0]
a = [[0,0], [1,1], [2,2]]

a = np.array(a)

a
a[0][1] = 1

a
X_res
score_v
#def auc_score(y_test,y_pred):

 #   return roc_auc_score(y_test,y_pred)



#kf = KFold(n_splits=4, shuffle=True, random_state=0)

#score_func = {'auc':make_scorer(auc_score)}

#scores = cross_validate(gscv, X_train, y_train, cv = kf, scoring=score_func)

#print('auc:', scores['test_auc'].mean())
#from sklearn.model_selection import cross_val_score

#scores = cross_val_score(gscv, X_valid, y_valid, cv=kf)

#scores.mean()
reg = LGBMClassifier()

ros = SMOTE(random_state=0)

X, y = ros.fit_sample(X_train, y_train)

reg.fit(X, y)
test = test_df.values

p = reg.predict_proba(test)
sample = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

sample['Diabetes'] = p[:,1]

sample.to_csv('pre.csv', index=False)