import numpy as np

import scipy as sp

import pandas as pd

import lightgbm as lgb

import os

import re

import gc



%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm.notebook import tqdm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# DataFrame 表示列数の上限変更

pd.set_option('display.max_columns', 100)



# Kaggle 環境か否かを取得

is_local_env = 'KAGGLE_URL_BASE' not in os.environ.keys()



# コア数取得

if is_local_env:

    CORE_NUM = os.environ['NUMBER_OF_PROCESSORS']

else:

    CORE_NUM = 1
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line']) # skiprows=lambda x: x%20!=0

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
# issue_d から年月情報を追加

df_train['year'] = df_train.issue_d.dt.year

df_train['month'] = df_train.issue_d.dt.month

df_test['year'] = df_test.issue_d.dt.year

df_test['month'] = df_test.issue_d.dt.month
# earliest_cr_line から年情報を追加

df_train['ecl_year'] = df_train.earliest_cr_line.dt.year

df_test['ecl_year'] = df_test.earliest_cr_line.dt.year
# サブテーブルのマージ - spi.csv

df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])

df_spi['year'] = df_spi.date.dt.year

df_spi['month'] = df_spi.date.dt.month



df_temp = df_spi.groupby(['year', 'month'], as_index=False)['close'].mean() # 年月で GroupBy 平均

df_train = df_train.merge(df_temp, on=['year', 'month'], how='left')

df_test = df_test.merge(df_temp, on=['year', 'month'], how='left')
fillna_param = {

#    'annual_inc' : 0,

#    'collections_12_mths_ex_med' : 0,

#    'acc_now_delinq' : 0,

#    'tot_coll_amt' : 0,

#    'tot_cur_bal' : 0,

#    'delinq_2yrs' : 0,

#    'open_acc' : 0,

#    'revol_bal' : 0,

#    'revol_util' : 0,

#    'pub_rec' : 0,

#    'inq_last_6mths' : 0,

    'mths_since_last_delinq' : 9999,

    'mths_since_last_record' : 9999,

    'mths_since_last_major_derog' : 9999,

    'emp_title': '#',

    'title': '#',

}
# 上記定義に沿った NaN 埋め

df_train = df_train.fillna(fillna_param)

df_test = df_test.fillna(fillna_param)



# 欠損値の補完 - 空白を中央値で埋める

# mean_cols = ['dti', 'revol_util', 'total_acc', 'ecl_year']



# for col in mean_cols:

#     median = df_train[col].median()

#     df_train[col] = [median if str(x) == 'nan' else x for x in df_train[col]]

#     df_test[col] = [median if str(x) == 'nan' else x for x in df_test[col]]



# 定義にないものは適当に埋める

df_train = df_train.fillna(-9999)

df_test = df_test.fillna(-9999)
ratio_cols = ['grade', 'sub_grade', 'purpose', 'addr_state']



# 算出は学習データで行い、同じ値をテストデータにも適応させる

for col in ratio_cols:

    all_cnt = df_train[col].value_counts()

    true_cnt = df_train.query('loan_condition == 1')[col].value_counts()

    ratio_values = true_cnt / all_cnt

    ratio_values = ratio_values.fillna(0)

    df_train['ratio_' + col] = df_train[col].map(ratio_values)

    df_test['ratio_' + col] = df_test[col].map(ratio_values)
# 特徴量追加 : xxx / yyy --> 各値の比率の特徴量

value_cols = [

    'loan_amnt',

    'dti',

    'annual_inc',

    'revol_bal',

    'revol_util',

    'open_acc',

    'close',

    'tot_cur_bal',

    'installment'

]



for x_col in value_cols:

    for y_col in value_cols:

        if x_col == y_col:

            continue

        df_train[x_col + '*' + y_col] = df_train[x_col] * df_train[y_col]

        df_train[x_col + '/' + y_col] = df_train[x_col] / df_train[y_col]

        df_test[x_col + '*' + y_col] = df_test[x_col] * df_test[y_col]

        df_test[x_col + '/' + y_col] = df_test[x_col] / df_test[y_col]
X_train = df_train.drop(['loan_condition'], axis=1)

y_train = df_train['loan_condition']

X_test = df_test
# grade, sub_grade は順序に意味を持たせるためセルフエンコード

# grade_cols = ['grade', 'sub_grade']

grade_cols = ['sub_grade']



def set_grade_value(grade):

    if grade == 'A':

        return 1

    if grade == 'B':

        return 2

    if grade == 'C':

        return 3

    if grade == 'D':

        return 4

    if grade == 'E':

        return 5

    if grade == 'F':

        return 6

    if grade == 'G':

        return 7

    

for col in grade_cols:

    unique = pd.unique(X_train[col])

    unique.sort()

    

    items = []

    indicies = []

    for i, item in enumerate(unique):

        items.append(item)

        grade_val = set_grade_value(item[0])

        rank_val = float(item[1]) / 10

        indicies.append(grade_val + rank_val)



    grade_vals = pd.Series(indicies, index=items)

    X_train[col] = X_train[col].map(grade_vals)

    X_test[col] = X_test[col].map(grade_vals)
# sub_grade で重みづけされた or 特徴量を追加する

g_weight_cols = [

    'loan_amnt',

    'dti',

    'annual_inc',

    'revol_bal',

    'revol_util',

    'open_acc',

    'close',

    'tot_cur_bal',

    'installment'

]



# for col in g_weight_cols:

#     X_train[col + '*sub_grade'] = X_train[col] * X_train['sub_grade']

#     X_test[col + '*sub_grade'] = X_test[col] * X_test['sub_grade']
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)



# あとでテキスト処理するので除外

cats.remove('emp_title')

cats.remove('title')



oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train
# このあたりからメモリを逼迫してくるので GC

del df_train, df_test

gc.collect()
%%time

txt_cols = ['emp_title', 'title']



for txt in txt_cols:

    tfidf = TfidfVectorizer(max_features=128, analyzer='word', ngram_range=(1, 2))

    txt_train = tfidf.fit_transform(X_train[txt])

    txt_test = tfidf.transform(X_test[txt])



    txt_train.todense()

    txt_test.todense()

    df_TXT_train = pd.DataFrame(txt_train.toarray(), columns=['txt_' + txt + '_' + name for name in tfidf.get_feature_names()])

    df_TXT_test = pd.DataFrame(txt_test.toarray(), columns=['txt_' + txt + '_' + name for name in tfidf.get_feature_names()])



    X_train = pd.concat([X_train, df_TXT_train], axis=1)

    X_test = pd.concat([X_test, df_TXT_test], axis=1)    
# del_cols = ['issue_d', 'title', 'earliest_cr_line', 'emp_title', 'year', 'month', 'City']

del_cols = ['grade','issue_d', 'title', 'earliest_cr_line', 'emp_title', 'year']



for col in del_cols:

    X_train.drop([col], axis=1, inplace=True)

    X_test.drop([col], axis=1, inplace=True)
display(X_train)

display(X_test)
del txt_train, txt_test, df_TXT_train, df_TXT_test

gc.collect()
# 交差検定してスコアを見てみる。層化抽出で良いかは別途よく検討。

if is_local_env:

    scores = []



    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        # clf = GradientBoostingClassifier()

        clf = lgb.LGBMClassifier(n_jobs=CORE_NUM)

        

        clf.fit(X_train_, y_train_)

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)



        print('CV Score of Fold_%d is %f' % (i, score))



    print('Mean : ', np.mean(scores))
# CV Averaging 実行

SPLIT_NUM = 5

df_cv_avg = pd.DataFrame() # Average 計算用 DataFrame



skf = StratifiedKFold(n_splits=SPLIT_NUM, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    # clf = GradientBoostingClassifier()

    clf = lgb.LGBMClassifier(n_jobs=CORE_NUM)

        

    clf.fit(X_train_, y_train_)

    # y_pred = clf.predict_proba(X_val)[:,1]

    y_pred = clf.predict_proba(X_test)[:,1]

    series = pd.Series(y_pred, name='rslt_' + str(i))

    df_cv_avg = pd.concat([df_cv_avg, series], axis=1)

    print('CV Score of Fold_%d is completed.' % (i))

 

    del X_train_, y_train_, X_val, y_val

    gc.collect()



df_cv_avg['rslt_avg'] = df_cv_avg.mean(axis=1)

display(df_cv_avg)
# 全データで再学習し、testに対して予測する

# clf = lgb.LGBMClassifier(n_jobs=CORE_NUM)

# clf.fit(X_train, y_train)

# y_pred = clf.predict_proba(X_test)[:,1]
imp = pd.DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance'])

sorted = imp.sort_values('importance', ascending=False)

display(sorted)



sorted.to_csv('feature_importance.csv')
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



df_cv_avg.set_index(submission.index, inplace=True)

submission.loan_condition = df_cv_avg['rslt_avg']

submission.to_csv('submission.csv')

submission