#!pip uninstall tensorflow -y

#!pip install tensorflow==1.11.0



#!pip uninstall keras -y

#!pip install keras==2.2.4
#from keras.layers import Input, Dense ,Dropout, BatchNormalization

#from keras.optimizers import Adam, SGD

#from keras.models import Model

#from keras.callbacks import EarlyStopping
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import quantile_transform



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d'])

df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d'])
X_train = df_train.drop(['loan_condition'], axis=1)

y_train = df_train.loan_condition



X_test = df_test
cats = []

nums = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

    else:

        nums.append(col)



#cats.remove('emp_title')

nums.remove('issue_d')
# 住所（州）と年をグループ識別子として分離しておく。

#groups = X_train.addr_state.values

#year = X_train.issue_d.dt.year



#未使用列削除

X_train.drop(['issue_d'], axis=1, inplace=True)

X_test.drop(['issue_d'], axis=1, inplace=True)
#欠損値(数値)_中央値

X_train['annual_inc'].fillna(X_train['annual_inc'].median(), inplace=True)

X_train['dti'].fillna(X_train['dti'].median(), inplace=True)

X_train['delinq_2yrs'].fillna(X_train['delinq_2yrs'].median(), inplace=True)

X_train['inq_last_6mths'].fillna(X_train['inq_last_6mths'].median(), inplace=True)

X_train['open_acc'].fillna(X_train['open_acc'].median(), inplace=True)

X_train['pub_rec'].fillna(X_train['pub_rec'].median(), inplace=True)

X_train['total_acc'].fillna(X_train['total_acc'].median(), inplace=True)

X_train['acc_now_delinq'].fillna(X_train['acc_now_delinq'].median(), inplace=True)

X_test['dti'].fillna(X_train['dti'].median(), inplace=True)

X_test['inq_last_6mths'].fillna(X_train['inq_last_6mths'].median(), inplace=True)

#欠損値(数値)_0

X_train['mths_since_last_delinq'].fillna(0, inplace=True)

X_train['mths_since_last_record'].fillna(0, inplace=True)

X_train['revol_util'].fillna(0, inplace=True)

X_train['collections_12_mths_ex_med'].fillna(0, inplace=True)

X_train['mths_since_last_major_derog'].fillna(0, inplace=True)

X_train['tot_coll_amt'].fillna(0, inplace=True)

X_train['tot_cur_bal'].fillna(0, inplace=True)

X_test['mths_since_last_delinq'].fillna(0, inplace=True)

X_test['mths_since_last_record'].fillna(0, inplace=True)

X_test['revol_util'].fillna(0, inplace=True)

X_test['mths_since_last_major_derog'].fillna(0, inplace=True)
#ランクガウス

X_all = pd.concat([X_train, X_test], axis=0)

X_all[nums] = quantile_transform(X_all[nums], n_quantiles=100, random_state=0, output_distribution='normal')
X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]
#カテゴリ欠損値処理

for col in cats:

    X_train[col].fillna("NULL", inplace=True)

    X_test[col].fillna("NULL", inplace=True)
#targetエンコーディング

target = 'loan_condition'



for col in cats:

    X_temp = pd.concat([X_train, y_train], axis=1)



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary) 

    

    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

    

    X_train[col] = enc_train

    X_test[col] = enc_test
#カテゴリ欠損値処理

for col in cats:

    X_train[col].fillna(0, inplace=True)

    X_test[col].fillna(0, inplace=True)
#層化抽出

scores = []



y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]

    

    clf = LGBMClassifier(max_depth=5, learning_rate=0.05,n_estimators=9999,colsample_bytree=0.7,reg_alpha=1,reg_lambda=1)

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(y_val, y_pred))

    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく

    

scores = np.array(scores)
print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 5
submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')