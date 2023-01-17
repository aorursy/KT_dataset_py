# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import scipy as sp

import numpy as np # linear algebra

from pandas import DataFrame, Series

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#データの読み込み

df_train = pd.read_csv("../input/train.csv", index_col = 0)

df_test = pd.read_csv("../input/test.csv", index_col = 0)
#効かなそうな特徴量を削除

df_train = df_train.drop('application_type', axis = 1)

df_train = df_train.drop('acc_now_delinq', axis = 1)

df_test = df_test.drop('application_type', axis = 1)

df_test = df_test.drop('acc_now_delinq', axis = 1)
#説明変数と目的変数にデータ分割

X_train = df_train.drop('loan_condition', axis = 1)

X_test = df_test

y_train = df_train.loan_condition
#行ごとの欠損値数を特徴量として追加

X_train['nan_sum'] = X_train.isnull().sum(axis = 1)

X_test['nan_sum'] = X_test.isnull().sum(axis = 1)
#カテゴリ型特徴量のエンジニアリング：One-hot Encoding

#cat_cols = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'addr_state', 'initial_list_status', 'application_type']

#cat_cols = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'initial_list_status']



#X_train = X_train[cat_cols]

X_train = pd.get_dummies(columns = ['grade'], data = X_train)

X_train = pd.get_dummies(columns = ['sub_grade'], data = X_train)

X_train = pd.get_dummies(columns = ['home_ownership'], data = X_train)

X_train = pd.get_dummies(columns = ['purpose'], data = X_train)

#X_train = pd.get_dummies(columns = ['addr_state'], data = X_train)

X_train = pd.get_dummies(columns = ['initial_list_status'], data = X_train)

#X_train = pd.get_dummies(columns = ['application_type'], data = X_train)



#X_test = X_test[cat_cols]

X_test = pd.get_dummies(columns = ['grade'], data = X_test)

X_test = pd.get_dummies(columns = ['sub_grade'], data = X_test)

X_test = pd.get_dummies(columns = ['home_ownership'], data = X_test)

X_test = pd.get_dummies(columns = ['purpose'], data = X_test)

#X_test = pd.get_dummies(columns = ['addr_state'], data = X_test)

X_test = pd.get_dummies(columns = ['initial_list_status'], data = X_test)

#X_test = pd.get_dummies(columns = ['application_type'], data = X_test)
#列数・列順を揃える

X_test_tmp = X_test #データ退避

X_test = pd.DataFrame(index=[])

#X_trainに存在しているが、X_testに存在しないカラムは全値0で列追加

for col in X_train.columns:

    if col in X_test_tmp.columns:

        X_test[col] = X_test_tmp[col]

    else:

        X_test[col] = int(0)



#X_testに存在しているが、X_trainに存在しないカラムは列削除        

#for col in X_test.columns:

#    if not col in X_train.columns:

#        X_test.drop([col], axis = 1, inplace = True)
#カテゴリ型特徴量のエンジニアリング：Count (Frequency) Encoding

summary = X_train.addr_state.value_counts()

X_train['addr_state'] = X_train.addr_state.map(summary)

summary = X_test.addr_state.value_counts()

X_test['addr_state'] = X_test.addr_state.map(summary)
#「emp_title」がNaNのデータにフラグを立てる

X_train.loc[X_train.emp_title.notnull(), 'emp_title_nanflg'] = 0

X_train['emp_title_nanflg'] = X_train['emp_title_nanflg'].fillna(1)

X_test.loc[X_test.emp_title.notnull(), 'emp_title_nanflg'] = 0

X_test['emp_title_nanflg'] = X_test['emp_title_nanflg'].fillna(1)
#テキスト型特徴量のエンジニアリング：TFIDF

#テキスト列のみ抜き出す

TXT1_train = X_train['emp_title']

TXT2_train = X_train['title']

TXT1_test = X_test['emp_title']

TXT2_test = X_test['title']
#欠損値補間

TXT1_train.fillna('#', inplace = True)

TXT2_train.fillna('#', inplace = True)

TXT1_test.fillna('#', inplace = True)

TXT2_test.fillna('#', inplace = True)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features = 100)

TXT1_train = tfidf.fit_transform(TXT1_train)

#X_train['emp_title'] = np.hstack(TXT1_test)

TXT1_test = tfidf.transform(TXT1_test)

#X_test['emp_title'] = np.hstack(TXT1_test)

"""

TXT2_train = tfidf.fit_transform(TXT2_train)

#X_train['title'] = np.hstack(TXT2_train)

TXT2_test = tfidf.transform(TXT2_test)

#X_test['title'] = np.hstack(TXT2_test)

"""
X_train_idx = X_train.reset_index()

TXT1_train2 = pd.DataFrame(TXT1_train.todense())

X_train = pd.concat([X_train_idx, TXT1_train2], axis = 1)

X_test_idx = X_test.reset_index()

TXT1_test2 = pd.DataFrame(TXT1_test.todense())

X_test = pd.concat([X_test_idx, TXT1_test2], axis = 1)

"""

X_train_idx = X_train.reset_index()

TXT2_train2 = pd.DataFrame(TXT2_train.todense())

X_train = pd.concat([X_train_idx, TXT2_train2], axis = 1)

X_test_idx = X_test.reset_index()

TXT2_test2 = pd.DataFrame(TXT2_test.todense())

X_test = pd.concat([X_test_idx, TXT2_test2], axis = 1)

"""
#テキスト列を削除する

X_train.drop(['emp_title' ,'title'], axis = 1, inplace = True)

X_test.drop(['emp_title' ,'title'], axis = 1, inplace = True)
#日付データから差分を算出

from datetime import datetime as dt

a = pd.to_datetime(X_train['issue_d'])

b = pd.to_datetime(X_train['earliest_cr_line'])

X_train['date_diff'] = abs(b-a).dt.days



a = pd.to_datetime(X_test['issue_d'])

b = pd.to_datetime(X_test['earliest_cr_line'])

X_test['date_diff'] = abs(b-a).dt.days
#日付の元データ列を削除する

X_train.drop(['issue_d' ,'earliest_cr_line'], axis = 1, inplace = True)

X_test.drop(['issue_d' ,'earliest_cr_line'], axis = 1, inplace = True)
#数値と文字列が混在しているデータから、数値のみ抜粋

X_train['emp_length'] = X_train['emp_length'].str.extract('([0-9]+)').astype(float)

X_train['zip_code'] = X_train['zip_code'].str.extract('([0-9]+)')

X_test['emp_length'] = X_test['emp_length'].str.extract('([0-9]+)').astype(float)

X_test['zip_code'] = X_test['zip_code'].str.extract('([0-9]+)')
#欠損を平均値で補間

X_train.loan_amnt = X_train.loan_amnt.fillna(X_train.loan_amnt.mean())

X_train.installment = X_train.installment.fillna(X_train.installment.mean())

X_train.emp_length = X_train.emp_length.fillna(X_train.emp_length.mean())

X_train.annual_inc = X_train.annual_inc.fillna(X_train.annual_inc.mean())

X_train.dti = X_train.dti.fillna(X_train.dti.mean())

X_train.delinq_2yrs = X_train.delinq_2yrs.fillna(X_train.delinq_2yrs.mean())

X_train.inq_last_6mths = X_train.inq_last_6mths.fillna(X_train.inq_last_6mths.mean())

X_train.mths_since_last_delinq = X_train.mths_since_last_delinq.fillna(X_train.mths_since_last_delinq.mean())

X_train.mths_since_last_record = X_train.mths_since_last_record.fillna(X_train.mths_since_last_record.mean())

X_train.open_acc = X_train.open_acc.fillna(X_train.open_acc.mean())

X_train.pub_rec = X_train.pub_rec.fillna(X_train.pub_rec.mean())

X_train.revol_bal = X_train.revol_bal.fillna(X_train.revol_bal.mean())

X_train.revol_util = X_train.revol_util.fillna(X_train.revol_util.mean())

X_train.total_acc = X_train.total_acc.fillna(X_train.total_acc.mean())

X_train.collections_12_mths_ex_med = X_train.collections_12_mths_ex_med.fillna(X_train.collections_12_mths_ex_med.mean())

X_train.mths_since_last_major_derog = X_train.mths_since_last_major_derog.fillna(X_train.mths_since_last_major_derog.mean())

#X_train.acc_now_delinq = X_train.acc_now_delinq.fillna(X_train.acc_now_delinq.mean())

X_train.tot_coll_amt = X_train.tot_coll_amt.fillna(X_train.tot_coll_amt.mean())

X_train.tot_cur_bal = X_train.tot_cur_bal.fillna(X_train.tot_cur_bal.mean())

X_train.date_diff = X_train.date_diff.fillna(X_train.date_diff.mean())

X_train.addr_state = X_train.addr_state.fillna(0)



X_test.loan_amnt = X_test.loan_amnt.fillna(X_test.loan_amnt.mean())

X_test.installment = X_test.installment.fillna(X_test.installment.mean())

X_test.emp_length = X_test.emp_length.fillna(X_test.emp_length.mean())

X_test.annual_inc = X_test.annual_inc.fillna(X_test.annual_inc.mean())

X_test.dti = X_test.dti.fillna(X_test.dti.mean())

X_test.delinq_2yrs = X_test.delinq_2yrs.fillna(X_test.delinq_2yrs.mean())

X_test.inq_last_6mths = X_test.inq_last_6mths.fillna(X_test.inq_last_6mths.mean())

X_test.mths_since_last_delinq = X_test.mths_since_last_delinq.fillna(X_test.mths_since_last_delinq.mean())

X_test.mths_since_last_record = X_test.mths_since_last_record.fillna(X_test.mths_since_last_record.mean())

X_test.open_acc = X_test.open_acc.fillna(X_test.open_acc.mean())

X_test.pub_rec = X_test.pub_rec.fillna(X_test.pub_rec.mean())

X_test.revol_bal = X_test.revol_bal.fillna(X_test.revol_bal.mean())

X_test.revol_util = X_test.revol_util.fillna(X_test.revol_util.mean())

X_test.total_acc = X_test.total_acc.fillna(X_test.total_acc.mean())

X_test.collections_12_mths_ex_med = X_test.collections_12_mths_ex_med.fillna(X_test.collections_12_mths_ex_med.mean())

X_test.mths_since_last_major_derog = X_test.mths_since_last_major_derog.fillna(X_test.mths_since_last_major_derog.mean())

#X_test.acc_now_delinq = X_test.acc_now_delinq.fillna(X_test.acc_now_delinq.mean())

X_test.tot_coll_amt = X_test.tot_coll_amt.fillna(X_test.tot_coll_amt.mean())

X_test.tot_cur_bal = X_test.tot_cur_bal.fillna(X_test.tot_cur_bal.mean())

X_test.date_diff = X_test.date_diff.fillna(X_test.date_diff.mean())

X_test.addr_state = X_test.addr_state.fillna(0)
#ビン分割

X_train['loan_amnt'] = pd.cut(X_train['loan_amnt'], [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000, 75000, 78000, 81000, 84000, 87000, 90000, 93000, 96000, 99000, 102000], labels = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000, 75000, 78000, 81000, 84000, 87000, 90000, 93000, 96000, 99000], right = False).astype(float)

X_test['loan_amnt'] = pd.cut(X_test['loan_amnt'], [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000, 75000, 78000, 81000, 84000, 87000, 90000, 93000, 96000, 99000, 102000], labels = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000, 42000, 45000, 48000, 51000, 54000, 57000, 60000, 63000, 66000, 69000, 72000, 75000, 78000, 81000, 84000, 87000, 90000, 93000, 96000, 99000], right = False).astype(float)
#X_train = X_train.drop('index', axis = 1)

X_train = X_train.drop('ID', axis = 1)

#X_test = X_test.drop('index', axis = 1)

X_test = X_test.drop('ID', axis = 1)
#カラム名取得

num_cols = []

for col in X_train.columns:

    num_cols.append(col)

    

print(num_cols)
X_train
#数値型特徴量のエンジニアリング：StandardScaler

from sklearn.preprocessing import StandardScaler

#num_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'zip_code', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'date_diff']

#num_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'zip_code', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'date_diff', 'addr_state']

scaler = StandardScaler()

scaler.fit(X_train[num_cols])
#変換後のデータで各列を置換

X_train[num_cols] = scaler.transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from lightgbm import LGBMClassifier
"""

%%time

# CVしてスコアを見てみる

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier() # ここではデフォルトのパラメータになっている。各自の検討項目です

    clf = LGBMClassifier()

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))

"""
#全データで再学習し、testに対して予測する

#clf = GradientBoostingClassifier() # ここではデフォルトのパラメータになっている。各自の検討項目です

clf = LGBMClassifier()

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する
#sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')