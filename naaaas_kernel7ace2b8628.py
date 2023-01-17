import numpy as np

import scipy as sp

import pandas as pd

import lightgbm as lgb

import os

import re



%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm.notebook import tqdm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_squared_log_error



# DataFrame 表示列数の上限変更

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 200)



# Kaggle 環境か否かを取得

is_local_env = 'KAGGLE_URL_BASE' not in os.environ.keys()



# コア数取得

if is_local_env:

    CORE_NUM = os.environ['NUMBER_OF_PROCESSORS']

else:

    CORE_NUM = 1
PATH_TRAIN = '../input/exam-for-students20200129/train.csv'

PATH_TEST  = '../input/exam-for-students20200129/test.csv'

PATH_SUBMISSION  = '../input/exam-for-students20200129/sample_submission.csv'

TARGET = 'ConvertedSalary'



# 読み込み時、時間データとしてパースする列名

COLS_PARSE_DATES = []



# テキスト特徴量として扱う列名

COLS_TEXT = ['FrameworkWorkedWith', 'DevType', 'CommunicationTools']



# 最終的に使わなくなる特徴量の列名

COLS_DELETE = ['Country']
# 学習・テストデータ読み込み

df_train = pd.read_csv(PATH_TRAIN, index_col=0, parse_dates=COLS_PARSE_DATES) # , skiprows=lambda x: x % 20 != 0

df_test = pd.read_csv(PATH_TEST, index_col=0, parse_dates=COLS_PARSE_DATES)

print('df_train:', df_train.shape)

print('df_test :', df_test.shape)
# 分布が Log のようになっているので、大部分を占める以外の大きく離れた値は学習データから抜いてしまう

df_train = df_train[df_train[TARGET] < 300000]

print('*under 300K* df_train:', df_train.shape)
# なぜか country_info の数値データの小数点がカンマになっているので要修正

# しかし、ピリオド小数点のものも存在するので、手で直す ⇒ https://qiita.com/kt38k/items/ddc42a6feda3eca6674e

df_info = pd.read_csv('../input/exam-for-students20200129/country_info.csv', decimal=',')

df_info['GDP ($ per capita)'] = df_info['GDP ($ per capita)'].astype(float)

display(df_info)

display([df_info[x].dtype for x in df_info.columns])



df_train = df_train.merge(df_info, on=['Country'], how='left')

df_test = df_test.merge(df_info, on=['Country'], how='left')
# テキスト特徴量の欠損値補完  文字列で置換

for txt in COLS_TEXT:

    df_train = df_train.fillna({txt: '(NAN)'})

    df_test = df_test.fillna({txt: '(NAN)'})
# 欠損値の補完  LightGBM なのでよしなに

df_train = df_train.fillna(-9999)

df_test = df_test.fillna(-9999)
X_train = df_train.drop([TARGET], axis=1)

y_train = df_train[TARGET]

X_test = df_test
# YearsCoding YearsCodingProf は順序に意味を持たせるためセルフエンコード

# grade_cols = ['grade', 'sub_grade']

years_cols = ['YearsCoding', 'YearsCodingProf']



def set_grade_value(grade):

    if grade == '0-2 years':

        return 2

    if grade == '3-5 years':

        return 3

    if grade == '6-8 years':

        return 4

    if grade == '9-11 years':

        return 5

    if grade == '12-14 years':

        return 6

    if grade == '15-17 years':

        return 7

    if grade == '18-20 years':

        return 8

    if grade == '21-23 years':

        return 9

    if grade == '24-26 years':

        return 10

    if grade == '27-29 years':

        return 11

    if grade == '30 or more years':

        return 12

    else: # == NULL

        return 99

    

for col in years_cols:

    unique = pd.unique(X_train[col])

    

    items = []

    indicies = []

    for i, item in enumerate(unique):

        items.append(item)

        grade_val = set_grade_value(item)

        indicies.append(grade_val)



    grade_vals = pd.Series(indicies, index=items)

    X_train[col] = X_train[col].map(grade_vals)

    X_test[col] = X_test[col].map(grade_vals)

    display(grade_vals)
# OrdinalEncoder によるエンコーディング

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)



# あとでテキスト処理する列はここで除外

for txt in COLS_TEXT:

    cats.remove(txt)



oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
# テキスト特徴量の生成

df_txt_train_list = []

df_txt_test_list = []



for txt in COLS_TEXT:

    df_txt_train = X_train[txt].copy()

    df_txt_test = X_test[txt].copy()



    tfidf = TfidfVectorizer(max_features=64, use_idf=True)

    TXT_train = tfidf.fit_transform(df_txt_train)

    TXT_test = tfidf.transform(df_txt_test)

    

    df_TXT_train = pd.DataFrame(TXT_train.toarray(), columns=tfidf.get_feature_names(), index=list(X_train.index))

    df_TXT_test = pd.DataFrame(TXT_test.toarray(), columns=tfidf.get_feature_names(), index=list(X_test.index))

    df_txt_train_list.append(df_TXT_train)

    df_txt_test_list.append(df_TXT_test)
for df in df_txt_train_list:

    X_train = pd.concat([X_train, df], axis=1)

for df in df_txt_test_list:

    X_test = pd.concat([X_test, df], axis=1)
# テキスト特徴量の元ネタ削除

for col in COLS_TEXT:

    X_train.drop([col], axis=1, inplace=True)

    X_test.drop([col], axis=1, inplace=True)
# 使わない特徴量の削除

for col in COLS_DELETE:

    X_train.drop([col], axis=1, inplace=True)

    X_test.drop([col], axis=1, inplace=True)
display(X_train)

display(X_test)
# 交差検定の実行（カーネルでは走らない）

if is_local_env:

    scores = []



    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        clf = lgb.LGBMRegressor(objective='regression', n_jobs=CORE_NUM)

        

        clf.fit(X_train_, y_train_)

        y_pred = clf.predict(X_val)

        

        # 負値があると RMSLE が計算できないので、全て学習データの最小値に置換

        print(len([x for x in y_pred if x < 0]))

        y_pred = [y_train.min() if x < 0 else x for x in y_pred]

        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))

        scores.append(rmsle)



        print('CV Score (RMSLE) of Fold_%d is %f' % (i, rmsle))



    print('Mean : ', np.mean(scores))
# 全データで再学習し、testに対して予測する

clf = lgb.LGBMRegressor(n_jobs=CORE_NUM)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# 負価は全て 学習データの最小値 に置換

print(len([x for x in y_pred if x < 0]))

y_pred = [y_train.min() if x < 0 else x for x in y_pred]
# 特徴量の重要性をチェック

imp = pd.DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance'])

sorted = imp.sort_values('importance', ascending=False)

display(sorted)

sorted.to_csv('feature_importance.csv')
# 結果出力

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0) # , skiprows=lambda x: x % 20 != 0

submission[TARGET] = y_pred

submission.to_csv('submission.csv')