# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install -U category_encoders

import category_encoders as ce



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler
df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)
# trainから列をドロップ

y_train = df_train.loan_condition

X_train = df_train.drop(['emp_title', 'issue_d', 'title', 'emp_length', 'earliest_cr_line', 'loan_condition'], axis=1)

X_test = df_test.drop(['emp_title', 'issue_d', 'title', 'emp_length', 'earliest_cr_line'], axis=1)
# 欠損処理(数値)

# 中央値で埋めるが、同時に欠損フラグを立てる

# 時間ができたらリファクタしたい

X_train['mths_since_last_delinq_nan_flg'] = 0

X_train.loc[ (X_train['mths_since_last_delinq']!=X_train['mths_since_last_delinq']), 'mths_since_last_delinq_nan_flg'] = 1

X_train['mths_since_last_delinq'].fillna(X_train['mths_since_last_delinq'].median(),inplace=True)

X_test['mths_since_last_delinq_nan_flg'] = 0

X_test.loc[ (X_test['mths_since_last_delinq']!=X_test['mths_since_last_delinq']), 'mths_since_last_delinq_nan_flg'] = 1

X_test['mths_since_last_delinq'].fillna(X_train['mths_since_last_delinq'].median(),inplace=True)



X_train['mths_since_last_record_nan_flg'] = 0

X_train.loc[ (X_train['mths_since_last_record']!=X_train['mths_since_last_record']), 'mths_since_last_record_nan_flg'] = 1

X_train['mths_since_last_record'].fillna(X_train['mths_since_last_record'].median(),inplace=True)

X_test['mths_since_last_record_nan_flg'] = 0

X_test.loc[ (X_test['mths_since_last_record']!=X_test['mths_since_last_record']), 'mths_since_last_record_nan_flg'] = 1

X_test['mths_since_last_record'].fillna(X_train['mths_since_last_record'].median(),inplace=True)



X_train['mths_since_last_major_derog_nan_flg'] = 0

X_train.loc[ (X_train['mths_since_last_major_derog']!=X_train['mths_since_last_major_derog']), 'mths_since_last_major_derog_nan_flg'] = 1

X_train['mths_since_last_major_derog'].fillna(X_train['mths_since_last_major_derog'].median(),inplace=True)

X_test['mths_since_last_major_derog_nan_flg'] = 0

X_test.loc[ (X_test['mths_since_last_major_derog']!=X_test['mths_since_last_major_derog']), 'mths_since_last_major_derog_nan_flg'] = 1

X_test['mths_since_last_major_derog'].fillna(X_train['mths_since_last_major_derog'].median(),inplace=True)



X_train['tot_coll_amt_nan_flg'] = 0

X_train.loc[ (X_train['tot_coll_amt']!=X_train['tot_coll_amt']), 'tot_coll_amt_nan_flg'] = 1

X_train['tot_coll_amt'].fillna(X_train['tot_coll_amt'].median(),inplace=True)

X_test['tot_coll_amt_nan_flg'] = 0

X_test.loc[ (X_test['tot_coll_amt']!=X_test['tot_coll_amt']), 'tot_coll_amt_nan_flg'] = 1

X_test['tot_coll_amt'].fillna(X_train['tot_coll_amt'].median(),inplace=True)



X_train['tot_cur_bal_nan_flg'] = 0

X_train.loc[ (X_train['tot_cur_bal']!=X_train['tot_cur_bal']), 'tot_cur_bal_nan_flg'] = 1

X_train['tot_cur_bal'].fillna(X_train['tot_cur_bal'].median(),inplace=True)

X_test['tot_cur_bal_nan_flg'] = 0

X_test.loc[ (X_test['tot_cur_bal']!=X_test['tot_cur_bal']), 'tot_cur_bal_nan_flg'] = 1

X_test['tot_cur_bal'].fillna(X_train['tot_cur_bal'].median(),inplace=True)



X_train['dti_nan_flg'] = 0

X_train.loc[ (X_train['dti']!=X_train['dti']), 'dti_nan_flg'] = 1

X_train['dti'].fillna(X_train['dti'].median(),inplace=True)

X_test['dti_nan_flg'] = 0

X_test.loc[ (X_test['dti']!=X_test['dti']), 'dti_nan_flg'] = 1

X_test['dti'].fillna(X_train['dti'].median(),inplace=True)



X_train['inq_last_6mths_nan_flg'] = 0

X_train.loc[ (X_train['inq_last_6mths']!=X_train['inq_last_6mths']), 'inq_last_6mths_nan_flg'] = 1

X_train['inq_last_6mths'].fillna(X_train['inq_last_6mths'].median(),inplace=True)

X_test['inq_last_6mths_nan_flg'] = 0

X_test.loc[ (X_test['inq_last_6mths']!=X_test['inq_last_6mths']), 'inq_last_6mths_nan_flg'] = 1

X_test['inq_last_6mths'].fillna(X_train['inq_last_6mths'].median(),inplace=True)



X_train['revol_util_nan_flg'] = 0

X_train.loc[ (X_train['revol_util']!=X_train['revol_util']), 'revol_util_nan_flg'] = 1

X_train['revol_util'].fillna(X_train['revol_util'].median(),inplace=True)

X_test['revol_util_nan_flg'] = 0

X_test.loc[ (X_test['revol_util']!=X_test['revol_util']), 'revol_util_nan_flg'] = 1

X_test['revol_util'].fillna(X_train['revol_util'].median(),inplace=True)



X_train['annual_inc_nan_flg'] = 0

X_train.loc[ (X_train['annual_inc']!=X_train['annual_inc']), 'annual_inc_nan_flg'] = 1

X_train['annual_inc'].fillna(X_train['annual_inc'].median(),inplace=True)

X_test['annual_inc_nan_flg'] = 0

X_test.loc[ (X_test['annual_inc']!=X_test['annual_inc']), 'annual_inc_nan_flg'] = 1

X_test['annual_inc'].fillna(X_train['annual_inc'].median(),inplace=True)



X_train['delinq_2yrs_nan_flg'] = 0

X_train.loc[ (X_train['delinq_2yrs']!=X_train['delinq_2yrs']), 'delinq_2yrs_nan_flg'] = 1

X_train['delinq_2yrs'].fillna(X_train['delinq_2yrs'].median(),inplace=True)

X_test['delinq_2yrs_nan_flg'] = 0

X_test.loc[ (X_test['delinq_2yrs']!=X_test['delinq_2yrs']), 'delinq_2yrs_nan_flg'] = 1

X_test['delinq_2yrs'].fillna(X_train['delinq_2yrs'].median(),inplace=True)



X_train['open_acc_nan_flg'] = 0

X_train.loc[ (X_train['open_acc']!=X_train['open_acc']), 'open_acc_nan_flg'] = 1

X_train['open_acc'].fillna(X_train['open_acc'].median(),inplace=True)

X_test['open_acc_nan_flg'] = 0

X_test.loc[ (X_test['open_acc']!=X_test['open_acc']), 'open_acc_nan_flg'] = 1

X_test['open_acc'].fillna(X_train['open_acc'].median(),inplace=True)



X_train['pub_rec_nan_flg'] = 0

X_train.loc[ (X_train['pub_rec']!=X_train['pub_rec']), 'pub_rec_nan_flg'] = 1

X_train['pub_rec'].fillna(X_train['pub_rec'].median(),inplace=True)

X_test['pub_rec_nan_flg'] = 0

X_test.loc[ (X_test['pub_rec']!=X_test['pub_rec']), 'pub_rec_nan_flg'] = 1

X_test['pub_rec'].fillna(X_train['pub_rec'].median(),inplace=True)



X_train['total_acc_nan_flg'] = 0

X_train.loc[ (X_train['total_acc']!=X_train['total_acc']), 'total_acc_nan_flg'] = 1

X_train['total_acc'].fillna(X_train['total_acc'].median(),inplace=True)

X_test['total_acc_nan_flg'] = 0

X_test.loc[ (X_test['total_acc']!=X_test['total_acc']), 'total_acc_nan_flg'] = 1

X_test['total_acc'].fillna(X_train['total_acc'].median(),inplace=True)



X_train['collections_12_mths_ex_med_nan_flg'] = 0

X_train.loc[ (X_train['collections_12_mths_ex_med']!=X_train['collections_12_mths_ex_med']), 'collections_12_mths_ex_med_nan_flg'] = 1

X_train['collections_12_mths_ex_med'].fillna(X_train['collections_12_mths_ex_med'].median(),inplace=True)

X_test['collections_12_mths_ex_med_nan_flg'] = 0

X_test.loc[ (X_test['collections_12_mths_ex_med']!=X_test['collections_12_mths_ex_med']), 'collections_12_mths_ex_med_nan_flg'] = 1

X_test['collections_12_mths_ex_med'].fillna(X_train['collections_12_mths_ex_med'].median(),inplace=True)



X_train['acc_now_delinq_nan_flg'] = 0

X_train.loc[ (X_train['acc_now_delinq']!=X_train['acc_now_delinq']), 'acc_now_delinq_nan_flg'] = 1

X_train['acc_now_delinq'].fillna(X_train['acc_now_delinq'].median(),inplace=True)

X_test['acc_now_delinq_nan_flg'] = 0

X_test.loc[ (X_test['acc_now_delinq']!=X_test['acc_now_delinq']), 'acc_now_delinq_nan_flg'] = 1

X_test['acc_now_delinq'].fillna(X_train['acc_now_delinq'].median(),inplace=True)
# 欠損処理(カテゴリ)

# 'NoData'で埋める

# そのうちリファクタする

X_train['home_ownership'].fillna('NoData',inplace=True)

X_test['home_ownership'].fillna('NoData',inplace=True)



X_train['zip_code'].fillna('NoData',inplace=True)

X_test['zip_code'].fillna('NoData',inplace=True)



X_train['addr_state'].fillna('NoData',inplace=True)

X_test['addr_state'].fillna('NoData',inplace=True)
# 対数化

X_train.loan_amnt = X_train.loan_amnt.apply(np.log1p)

X_train.annual_inc = X_train.annual_inc.apply(np.log1p)

X_train.open_acc = X_train.open_acc.apply(np.log1p)

X_train.revol_bal = X_train.revol_bal.apply(np.log1p)

X_train.total_acc = X_train.total_acc.apply(np.log1p)

X_train.tot_cur_bal = X_train.tot_cur_bal.apply(np.log1p)



X_test.loan_amnt = X_test.loan_amnt.apply(np.log1p)

X_test.annual_inc = X_test.annual_inc.apply(np.log1p)

X_test.open_acc = X_test.open_acc.apply(np.log1p)

X_test.revol_bal = X_test.revol_bal.apply(np.log1p)

X_test.total_acc = X_test.total_acc.apply(np.log1p)

X_test.tot_cur_bal = X_test.tot_cur_bal.apply(np.log1p)
# 正規化

nums = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal']



# StandardScaler

scaler = StandardScaler()



X_train[nums] = X_train[nums].astype(float)

X_test[nums] = X_test[nums].astype(float)

scaler.fit(X_train[nums])



X_train[nums] = scaler.transform(X_train[nums])

X_test[nums] = scaler.transform(X_test[nums])
# カテゴリ値のエンコーディング

cats = ['grade', 'sub_grade', 'home_ownership', 'purpose', 'zip_code', 'addr_state', 'initial_list_status', 'application_type']



# ターゲットエンコーディング

ce_oe = ce.TargetEncoder(cols=cats)

X_oe = pd.concat([X_train, X_test], axis=0)



X_train[cats] = ce_oe.fit_transform(X_train[cats], y_train)

X_test[cats] = ce_oe.transform(X_test[cats])
# テキスト特徴量追加(title) とりあえずスパースなままくっつけてみる

import gc

from sklearn.feature_extraction.text import TfidfVectorizer



TXT_train = df_train['emp_title']

TXT_test = df_test['emp_title']



TXT_train.fillna('#', inplace=True)

TXT_test.fillna('#', inplace=True)



tfidf = TfidfVectorizer(max_features=200)

TXT_train = tfidf.fit_transform(TXT_train)

TXT_test = tfidf.transform(TXT_test)



gc.collect()

TXT_train_pd = pd.DataFrame(TXT_train.toarray(), columns=tfidf.get_feature_names())



gc.collect()

TXT_test_pd = pd.DataFrame(TXT_test.toarray(), columns=tfidf.get_feature_names())



df_train_ind = pd.read_csv('../input/train.csv', usecols=[0])

df_test_ind = pd.read_csv('../input/test.csv', usecols=[0])



TXT_train_pd = TXT_train_pd.set_index(df_train_ind['ID'])

TXT_test_pd = TXT_test_pd.set_index(df_test_ind['ID'])



gc.collect()

X_train = pd.concat([X_train, TXT_train_pd], axis=1)

X_test = pd.concat([X_test, TXT_test_pd], axis=1)
# Light GBM

import lightgbm as lgb

from lightgbm import LGBMClassifier



learning_rate = 0.1

num_leaves = 15

min_data_in_leaf = 2000

feature_fraction = 0.6

num_boost_round = 10000

params = {"objective": 'binary',

          "boosting_type": "gbdt",

          "learning_rate": learning_rate,

          "num_leaves": num_leaves,

          "feature_fraction": feature_fraction,

          "verbosity": 0,

          "drop_rate": 0.1,

          "is_unbalance": False,

          "max_drop": 50,

          "min_child_samples": 10,

          "min_child_weight": 150,

          "min_split_gain": 0,

          "subsample": 0.9,

          "metric": "auc",

          "boost_from_average": False

          }



NFOLDS = 5

skf = StratifiedKFold(n_splits=NFOLDS, random_state=71, shuffle=True)



final_cv_train = np.zeros(len(y_train))

final_cv_pred = np.zeros(len(X_test))



gc.collect()

for s in range(8):

    cv_train = np.zeros(len(y_train))

    cv_pred = np.zeros(len(X_test))



    params['seed'] = s



    kf = skf.split(X_train, y_train)



    best_trees = []

    fold_scores = []



    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        dtrain = lgb.Dataset(X_train_, y_train_)

        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,

                        early_stopping_rounds=200)

        best_trees.append(bst.best_iteration)

        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)

        cv_train[test_ix] += bst.predict(X_val)



        score = roc_auc_score(y_val, cv_train[test_ix])

        print(score)

        fold_scores.append(score)



    cv_pred /= NFOLDS

    final_cv_train += cv_train

    final_cv_pred += cv_pred



    print("cv score:")

    print(roc_auc_score(y_train, cv_train))

    print("current score:", roc_auc_score(y_train, final_cv_train / (s + 1.)), s+1)

    print(fold_scores)

    print(best_trees, np.mean(best_trees))



print(final_cv_pred / 8.)
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

print(final_cv_pred / 8.)

submission['loan_condition'] = final_cv_pred / 8.

submission.to_csv('submission2-txt.csv')