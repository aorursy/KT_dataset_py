import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



# Encord

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, quantile_transform

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



# Model

from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier



# Validation

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.metrics import roc_auc_score
# 時系列として有用な項目を日付型として読み込み

df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

# df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, skiprows=lambda x: x%20!=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])



# df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, skiprows=lambda x: x%20!=0)

# #df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0)

# df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0)
df_train_1 = df_train[df_train['issue_d'].dt.year >= 2014]

df_train_1['year'] = df_train['issue_d'].dt.year

df_test_1 = df_test

df_test_1['year'] = df_test['issue_d'].dt.year
# # 追加データを読み込み -> 追加した結果、数値低下

# gdp = pd.read_csv('../input/homework-for-students4plus/US_GDP_by_State.csv', names=['city', 'locale', 'gross', 'growth', 'population', 'year'], header=1)

# # gdp = gdp[gdp.year==2015] # 直近1年分のデータのみに限定して取得

# # gdp.drop(['year'], axis=1, inplace=True)



# sl = pd.read_csv('../input/homework-for-students4plus/statelatlong.csv',names=['addr_state', 'latitude', 'longitude', 'city'], header=1)



# # 追加データ同士を結合

# sl_gdp = pd.merge(sl, gdp, on=['city'], how='left')
# # statelatlong, US_GDP_by_State を結合

# df_train_2 = pd.merge(df_train_1, sl_gdp, on=['addr_state','year'], how='left')

# df_test_2 = pd.merge(df_test_1, sl_gdp, on=['addr_state', 'year'], how='left')
# X,y へ分割

y_train = []

X_train = []



# y_train = df_train_2.loan_condition

# X_train = df_train_2.drop(['loan_condition'], axis=1)

# X_test = df_test_2



y_train = df_train_1.loan_condition

X_train = df_train_1.drop(['loan_condition'], axis=1)

X_test = df_test_1



X_all = pd.concat([X_train,X_test], axis=0)
# grade の序列をマッピング

X_all=X_all.replace({'grade':{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}})

X_all['grade'] = X_all['grade'].astype('object')

X_all.dtypes
# subgrade の序列をマッピング

X_all=X_all.replace({'sub_grade':{'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                                      'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,

                                      'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,

                                      'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,

                                      'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,

                                      'F1':26,'F2':27,'F3':28,'F4':29,'F5':30,

                                      'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}})

X_all['sub_grade'] = X_all['sub_grade'].astype('object')
# emp_length の序列をマッピング

X_all=X_all.replace({'emp_length':{'10+ years':1,'9 years':2,'8 years':3,'7 years':4,'6 years':5,

                                      '5 years':6,'4 years':7,'3 years':8,'2 years':9,'1 year':10,'< 1 year':11}})

X_all['emp_length'] = X_all['emp_length'].astype('object')
# 日付項目を数値項目に変換

X_all['issue_d'] = X_all['issue_d'].map(pd.Timestamp.timestamp).astype(int)

X_all['earliest_cr_line'] = X_all['earliest_cr_line'].map(pd.Timestamp.timestamp).astype(int)

# 申請から融資月までの期間を保持

X_all['processing_d'] = X_all['issue_d'] - X_all['earliest_cr_line']
X_all.dtypes
# 各カラムでの欠損値の有無を保持

for col in X_all.columns:

    X_all[col+"_null"] = (X_all[col].isnull()*1).astype('object')
# 特徴量タイプごとに分割する

cat = []

num = []



for col in X_all.columns:

    if X_all[col].dtype == 'object':

        if X_all[col].nunique() != 1: # ユニーク値しかないデータは除外

            cat.append(col)

            print(col, X_all[col].nunique())

    else:

        if col != 'issue_d': # issue_d は未来予測で不要のため除外

            num.append(col)



cat.remove('emp_title')

cat.remove('title')

cat.remove('zip_code')



cat_all = X_all[cat]

num_all = X_all[num].fillna(-9999)



emp_title_all = X_all.emp_title

title_all = X_all.title
cat_all
#　金額系の特徴量として各比率計算結果を追加

num_all['loan_inst'] = num_all['loan_amnt'] / num_all['installment']

num_all['loan_inc'] = num_all['loan_amnt'] / num_all['annual_inc']

num_all['loan_revol'] = num_all['loan_amnt'] / num_all['revol_bal']

num_all['loan_coll'] = num_all['loan_amnt'] / num_all['tot_coll_amt']

num_all['loan_cur'] = num_all['loan_amnt'] / num_all['tot_cur_bal']



num_all['inst_inc'] = num_all['installment'] / num_all['annual_inc']

num_all['inst_revol'] = num_all['installment'] / num_all['revol_bal']

num_all['inst_coll'] = num_all['installment'] / num_all['tot_coll_amt']

num_all['inst_cur'] = num_all['installment'] / num_all['tot_cur_bal']



num_all['inc_revol'] = num_all['annual_inc'] / num_all['revol_bal']

num_all['inc_coll'] = num_all['annual_inc'] / num_all['tot_coll_amt']

num_all['inc_cur'] = num_all['annual_inc'] / num_all['tot_cur_bal']



num_all['revol_coll'] = num_all['revol_bal'] / num_all['tot_coll_amt']

num_all['revol_cur'] = num_all['revol_bal'] / num_all['tot_cur_bal']



num_all['coll_cur'] = num_all['tot_coll_amt'] / num_all['tot_cur_bal']



num_all['monthly_free_money'] = (num_all['annual_inc'] / 12 * num_all['dti'] / 100 + num_all['tot_cur_bal']) - num_all['installment']



# 0値による除算エラーへ対応

num_all.replace([np.inf, -np.inf], np.nan, inplace=True)



num_all.describe()
#　RankGauss

num_all[num] = quantile_transform(num_all[num], n_quantiles=100, random_state=0, output_distribution='normal')

num_all
# Ordinal Encoding

for col in tqdm(cat):

    oe = OrdinalEncoder(return_df=False)



    cat_all[col] = oe.fit_transform(cat_all[col])
count_cat = ['grade',

 'sub_grade',

 'emp_length',

 'home_ownership',

 'purpose',

 'addr_state',

 'initial_list_status',

 'application_type']
# 各カラムでの欠損値の有無を保持

for col in tqdm(count_cat):

    summary = cat_all[col].value_counts()

    cat_all[col+"_cnt"] = cat_all[col].map(summary).astype('object')



cat_all
X_all = pd.concat([num_all, cat_all], axis=1) # numericにconcatしていく
X_all
# emp_title

emp_title_all.fillna('#', inplace=True)

tfidf1 = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300, ngram_range=(1,2))

emp_title_all = tfidf1.fit_transform(emp_title_all.fillna('#'))
# # title

# title_all.fillna('#', inplace=True)

# tfidf2 = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300, ngram_range=(1,2))

# title_all = tfidf2.fit_transform(title_all.fillna('#'))
X_all = pd.concat([X_all, pd.DataFrame(emp_title_all.todense(), index=X_all.index)], axis=1)

# X_all = pd.concat([X_all, pd.DataFrame(title_all.todense(), index=X_all.index)], axis=1)
X_all.fillna(-9999, inplace=True)
# トレーニングデータ・テストデータに分割

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]



del X_all

gc.collect()
# 学習用と検証用に分割する

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)



clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                     importance_type='split', learning_rate=0.05, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                     random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)
# 学習

# %%time

clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
# 特徴量インパクト集計

imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# 特徴量インパクトをしきい値としてカラムを限定

use_col = imp[imp['importance'] > 50].index

# use_col = imp.index[:164] # 変数重要度で特徴量を絞り込んでみましょう

use_col
# トレーニングデータ・テストデータに分割

X_train = X_train[use_col]

X_test = X_test[use_col]
# from hyperopt import fmin, tpe, hp, rand, Trials



# # 目的関数

# def objective(space):

#     scores = []



#     skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



#     for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#         X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#         X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



#         clf = LGBMClassifier(n_estimators=9999, **space) 



#         clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

#         y_pred = clf.predict_proba(X_val)[:,1]

#         score = roc_auc_score(y_val, y_pred)

#         scores.append(score)

        

#     scores = np.array(scores)

#     print(scores.mean())

    

#     return -scores.mean()
# # 探索空間を指定する。

# space ={

#         'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

#         'subsample': hp.uniform ('subsample', 0.8, 1),

#         'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),

#         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

#     }
# # 探索結果の格納先

# trials = Trials()



# best = fmin(fn=objective,

#               space=space, 

#               algo=tpe.suggest,

#               max_evals=20, 

#               trials=trials, 

#               rstate=np.random.RandomState(71) 

#              )



# trials = Trials()
# # 最適化実行

# best = fmin(fn=objective,

# space=space,

# algo=tpe.suggest,

# max_evals=100, trials=trials,

# rstate=np.random.RandomState(71)

# )
# # 結果の出力

# clf = LGBMClassifier(**best)

# print(clf)
# # LightGBM + Random seed averaging + CV averaging

# scores = []



# for j in range(42,73):

#     skf = StratifiedKFold(n_splits=5, random_state=j, shuffle=True)

#     y_tests = np.zeros(len(X_test.index))



#     clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

#                          importance_type='split', learning_rate=0.05, max_depth=-1,

#                          min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                          n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#                          random_state=j, reg_alpha=1.0, reg_lambda=1.0, silent=True,

#                          subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



#     for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#         X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#         X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



#         clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

#         y_pred = clf.predict_proba(X_val)[:,1]

#         score = roc_auc_score(y_val, y_pred)

#         scores.append(score)

#         y_tests += clf.predict_proba(X_test)[:,1]
# LightGBM

scores = []



skf = StratifiedKFold(n_splits=5, random_state=72, shuffle=True)

y_tests = np.zeros(len(X_test.index))



clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                     importance_type='split', learning_rate=0.05, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                     random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    y_tests += clf.predict_proba(X_test)[:,1]
np.mean(scores)
y_pred  = y_tests/len(scores)
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')