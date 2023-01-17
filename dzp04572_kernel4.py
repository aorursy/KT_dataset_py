import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

import gc

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

from lightgbm import LGBMClassifier
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d'])

gc.collect()
df_train = df_train[df_train['delinq_2yrs'].isnull()==False]



f = lambda x : '1' if np.isnan(x) == True else '0'

f_train = pd.DataFrame(df_train['tot_coll_amt'].apply(f))

f_train = f_train.rename(columns={'tot_coll_amt':'tot_coll_amt_nan'})

df_train = pd.concat([df_train, f_train], axis=1)

f_test = pd.DataFrame(df_test['tot_coll_amt'].apply(f))

f_test = f_test.rename(columns={'tot_coll_amt':'tot_coll_amt_nan'})

df_test = pd.concat([df_test, f_test], axis=1)



df_train['zip_code'] = df_train['zip_code'].str[:3]

df_test['zip_code'] = df_test['zip_code'].str[:3]



df_train_label = df_train.index

df_test_label = df_test.index



# print(df_train.isnull().any())
pd.set_option('display.max_columns', 50)

# df_train
df_train = df_train.fillna({'annual_inc':df_train['annual_inc'].median(), 'dti':df_train['dti'].median(), 'tot_coll_amt':df_train['tot_coll_amt'].median(), 'tot_cur_bal':df_train['tot_cur_bal'].median(), 'open_acc':df_train['open_acc'].median(), 'total_acc':df_train['total_acc'].median(), 'inq_last_6mths':df_train['inq_last_6mths'].median(), 'revol_util':df_train['revol_util'].median(), 'mths_since_last_record':df_train['mths_since_last_record'].median()})

df_test = df_test.fillna({'annual_inc':df_train['annual_inc'].median(), 'dti':df_train['dti'].median(), 'tot_coll_amt':df_train['tot_coll_amt'].median(), 'tot_cur_bal':df_train['tot_cur_bal'].median(), 'open_acc':df_train['open_acc'].median(), 'total_acc':df_train['total_acc'].median(), 'inq_last_6mths':df_train['inq_last_6mths'].median(), 'revol_util':df_train['revol_util'].median(), 'mths_since_last_record':df_train['mths_since_last_record'].median()})
df_train = df_train.fillna({'delinq_2yrs':0, 'mths_since_last_delinq': 9999999, 'mths_since_last_major_derog': 9999999, 'pub_rec':0, 'collections_12_mths_ex_med':0, 'acc_now_delinq':0})

df_test = df_test.fillna({'delinq_2yrs':0, 'mths_since_last_delinq': 9999999, 'mths_since_last_major_derog': 9999999, 'pub_rec':0, 'collections_12_mths_ex_med':0, 'acc_now_delinq':0})
cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, df_train[col].nunique())
oe = OrdinalEncoder()

# dic = ['emp_length', 'emp_title', 'issue_d', 'title', 'earliest_cr_line']

dic = ['emp_length', 'emp_title', 'title', 'zip_code', 'earliest_cr_line']



oe_train = pd.DataFrame(oe.fit_transform(df_train[dic]), columns=dic)

oe_test = pd.DataFrame(oe.transform(df_test[dic]), columns=dic)



df_train['emp_length'] = oe_train['emp_length'].values

df_train['emp_title'] = oe_train['emp_title'].values

# df_train['issue_d'] = oe_train['issue_d'].values

df_train['zip_code'] = oe_train['zip_code'].values

df_train['title'] = oe_train['title'].values

df_train['earliest_cr_line'] = oe_train['earliest_cr_line'].values

df_test['emp_length'] = oe_test['emp_length'].values

df_test['emp_title'] = oe_test['emp_title'].values

# df_test['issue_d'] = oe_test['issue_d'].values

df_test['zip_code'] = oe_test['zip_code'].values

df_test['title'] = oe_test['title'].values

df_test['earliest_cr_line'] = oe_test['earliest_cr_line'].values
df_train['loan_amnt'] = df_train['loan_amnt'].apply(np.log1p)

df_train['annual_inc'] = df_train['annual_inc'].apply(np.log1p)

df_train['revol_bal'] = df_train['revol_bal'].apply(np.log1p)

df_train['tot_cur_bal'] = df_train['tot_cur_bal'].apply(np.log1p)

df_test['loan_amnt'] = df_test['loan_amnt'].apply(np.log1p)

df_test['annual_inc'] = df_test['annual_inc'].apply(np.log1p)

df_test['revol_bal'] = df_test['revol_bal'].apply(np.log1p)

df_test['tot_cur_bal'] = df_test['tot_cur_bal'].apply(np.log1p)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

dic2 = ['loan_amnt', 'annual_inc', 'revol_bal', 'tot_cur_bal']

sc_train = pd.DataFrame(sc.fit_transform(df_train[dic2]), columns=dic2)

sc_test = pd.DataFrame(sc.transform(df_test[dic2]), columns=dic2)
df_train = df_train.reset_index(drop=True)

df_train['loan_amnt'] = sc_train['loan_amnt'].values

df_train['revol_bal'] = sc_train['revol_bal'].values

df_train['tot_cur_bal'] = sc_train['tot_cur_bal'].values

df_test = df_test.reset_index(drop=True)

df_test['loan_amnt'] = sc_test['loan_amnt'].values

df_test['revol_bal'] = sc_test['revol_bal'].values

df_test['tot_cur_bal'] = sc_test['tot_cur_bal'].values
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)

X_test = df_test
col = 'grade'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary)

enc_test2 = pd.DataFrame(enc_test)

enc_test2.rename(columns={'grade':'enc_grade'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

enc_train2 = pd.DataFrame(enc_train)

enc_train2.rename(columns=lambda s:'enc_grade', inplace=True)
col = 'sub_grade'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary)

enc_test3 = pd.DataFrame(enc_test)

enc_test3.rename(columns={'sub_grade':'enc_sub_grade'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

enc_train3 = pd.DataFrame(enc_train)

enc_train3.rename(columns=lambda s:'enc_sub_grade', inplace=True)
col = 'home_ownership'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary)

enc_test4 = pd.DataFrame(enc_test)

enc_test4.rename(columns={'home_ownership':'enc_home_ownership'}, inplace=True)



# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



enc_train = Series(np.zeros(len(X_train)), index=df_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)

enc_train4 = pd.DataFrame(enc_train)

enc_train4.rename(columns=lambda s:'enc_home_ownership', inplace=True)
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        if col != 'issue_d':

            cats.append(col)

            print(col, X_train[col].nunique())

# del cats[4]

# del cats[3]

del cats[2]

del cats[1]

del cats[0]

cats
# year_train = X_train['issue_d'].dt.year

# month_train = X_train['issue_d'].dt.month

# year_test = X_test['issue_d'].dt.year

# month_test = X_test['issue_d'].dt.month

groups = X_train['issue_d'].values
ohe = OneHotEncoder(cols=cats, return_df=False)

ohe_train = pd.DataFrame(ohe.fit_transform(X_train[cats]))

ohe_test = pd.DataFrame(ohe.transform(X_test[cats]))
X_train2 = pd.concat([ohe_train, X_train], axis=1)

X_test2 = pd.concat([ohe_test, X_test], axis=1)

X_train3 = pd.concat([enc_train4, X_train2], axis=1)

X_test3 = pd.concat([enc_test4, X_test2], axis=1)

X_train4 = pd.concat([enc_train3, X_train3], axis=1)

X_test4 = pd.concat([enc_test3, X_test3], axis=1)

X_train5 = pd.concat([enc_train2, X_train4], axis=1)

X_test5 = pd.concat([enc_test2, X_test4], axis=1)
del X_train5['grade']

del X_train5['sub_grade']

del X_train5['home_ownership']

del X_train5['issue_d']

del X_train5['purpose']

del X_train5['addr_state']

del X_train5['initial_list_status']

del X_train5['application_type']

del X_train5['tot_coll_amt_nan']

del X_test5['grade']

del X_test5['sub_grade']

del X_test5['home_ownership']

del X_test5['issue_d']

del X_test5['purpose']

del X_test5['addr_state']

del X_test5['initial_list_status']

del X_test5['application_type']

del X_test5['tot_coll_amt_nan']
X_train0 = X_train5.set_index(df_train_label)

X_test0 = X_test5.set_index(df_test_label)
X_train = X_train0

X_test = X_test0
X_train['sub_grade_x_loan_amnt'] = X_train['enc_sub_grade'].values * X_train['loan_amnt'].values

X_test['sub_grade_x_loan_amnt'] = X_test['enc_sub_grade'].values * X_test['loan_amnt'].values
print(X_train.isnull().any())
# from hyperopt import fmin, tpe, hp, rand, Trials

# from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import roc_auc_score



# from lightgbm import LGBMClassifier
# def objective(space):

#     gkf = GroupKFold(n_splits=3)

#     scores = []

#     for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

#         X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#         X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

#         clf = LGBMClassifier(n_estimators=100, **space) 

#         clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

#         y_pred = clf.predict_proba(X_val)[:,1]

#         score = roc_auc_score(y_val, y_pred)

#         scores.append(score)       

#     scores = np.array(scores)

#     print(scores.mean())

  

#     return -scores.mean()
# space ={

#         'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

#         'subsample': hp.uniform ('subsample', 0.8, 1),

#         'learning_rate' : hp.quniform('learning_rate', 0.05, 0.1, 0.025),

#         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

# }
# trials = Trials()



# best = fmin(fn=objective,

#               space=space, 

#               algo=tpe.suggest,

#               max_evals=10, 

#               trials=trials, 

#               rstate=np.random.RandomState(71) 

#              )
# LGBMClassifier(**best)
# trials.best_trial['result']
# trials.best_trial
# %%time

# gkf = GroupKFold(n_splits=3)

# scores = []

# for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

#     X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#     X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

#     print('Train Groups', np.unique(groups_train_))

#     print('Val Groups', np.unique(groups_val))

#     clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

#                             importance_type='split', learning_rate=0.05, max_depth=-1,

#                             min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                             n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                             random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                             subsample=0.935, subsample_for_bin=200000, subsample_freq=0)

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

#     print('CV Score of Fold_%d is %f' % (i, score))

#     print('\n')
%%time

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,

                        importance_type='split', learning_rate=0.05, max_depth=-1,

                        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                        n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                        random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                        subsample=0.935, subsample_for_bin=200000, subsample_freq=0)

gkf = GroupKFold(n_splits=5)

scores = []

for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):   

    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

    print('Train Groups', np.unique(groups_train_))

    print('Val Groups', np.unique(groups_val))

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))

    print('\n')
print(np.mean(scores))

print(scores)
# %%time

# clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance'])
fig, ax = plt.subplots(figsize=(14, 14))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
y_pred = clf.predict_proba(X_test)[:,1]

y_pred
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred

submission.to_csv('submission.csv')