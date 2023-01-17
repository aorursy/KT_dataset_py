# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import category_encoders as ce

import re

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer



import lightgbm as lgb

from lightgbm import LGBMClassifier



from scipy import sparse, hstack



# datetime

import datetime



import numpy as np

import matplotlib.pyplot as plt



# 以下はこのノートブックのみ



# Stop Words: NLTKから英語のstop wordsを読み込み

from nltk.corpus import stopwords



# 可視化のためのセットです。

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
# 読み込み

df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)
# 読み込んだデータの確認(df_trainの始めの5行)

df_train.head()
# 読み込んだデータの確認(df_testの始めの5行)

df_test.head()
# trainデータをX, yへ分離

df_X_train = df_train.drop('loan_condition', axis=1)

df_y_train = df_train[['loan_condition']]

# testデータは元々Xのみだが、名前だけ変えておきます。

df_X_test = df_test
# X_trainを確認(始めの5行)

df_X_train.head()
# y_trainを確認(始めの5行)

df_y_train.head()
# X_testを確認(始めの5行)

df_X_test.head()
# 各カラムのdtypeを確認(ユニークなもの)

print('Trainデータのdtyep', df_X_train.dtypes.unique())

print('Testデータのdtype', df_X_test.dtypes.unique())
# 数値カラムを抽出 (float or int)

list_cols_num = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'float64' or df_X_train[i].dtype == 'int64':

        list_cols_num.append(i)

        

print(list_cols_num)
# 各統計量を作成

# X_train

statics_X_train_num = pd.DataFrame([df_X_train[list_cols_num].nunique().values.tolist(),  # ユニーク数

                                df_X_train[list_cols_num].isnull().sum().values.tolist(),  # 欠損数

                              df_X_train[list_cols_num].mean().values.tolist(),  # 平均値

                              df_X_train[list_cols_num].std().values.tolist(),  # 標準偏差

                              df_X_train[list_cols_num].median().values.tolist(),  # 中央値

                              df_X_train[list_cols_num].min().values.tolist(),  # 最小値

                              df_X_train[list_cols_num].max().values.tolist()],  # 最大値

                              index=['unique', 'null', 'mean', 'std', 'median', 'min', 'max'],

                              columns=list_cols_num).T

# X_test

statics_X_test_num =  pd.DataFrame([df_X_test[list_cols_num].nunique().values.tolist(),  # ユニーク数

                                df_X_test[list_cols_num].isnull().sum().values.tolist(),  # 欠損数

                                df_X_test[list_cols_num].mean().values.tolist(),  # 平均値

                                df_X_test[list_cols_num].std().values.tolist(),  # 標準偏差

                                df_X_test[list_cols_num].median().values.tolist(),  # 中央値

                                df_X_test[list_cols_num].min().values.tolist(),  # 最小値

                                df_X_test[list_cols_num].max().values.tolist()],  # 最大値

                                index=['unique', 'null', 'mean', 'std', 'median', 'min', 'max'],

                                columns=list_cols_num).T

# y_traine

statics_y_train_num = pd.DataFrame([df_y_train['loan_condition'].astype('float64').nunique(),  # ユニーク数

                                df_y_train['loan_condition'].astype('float64').isnull().sum(),  # 欠損数

                                df_y_train['loan_condition'].astype('float64').mean(),  # 平均値

                                df_y_train['loan_condition'].astype('float64').std(),  # 標準偏差

                                df_y_train['loan_condition'].astype('float64').median(),  # 中央値

                                df_y_train['loan_condition'].astype('float64').min(),  # 最小値

                                df_y_train['loan_condition'].astype('float64').max()],  # 最大値

                                index=['unique', 'null', 'mean', 'std', 'median', 'min', 'max'],

                                columns=['loan_condition']).T
statics_X_train_num
statics_X_test_num
statics_y_train_num
# ヒストグラムを描いてみる

plotpos = 1

fig = plt.figure(figsize = (30, 30))



for i in list_cols_num:

    plotdata1 = df_X_train[i]

    plotdata2 = df_X_test[i]



    ax = fig.add_subplot(6, 3, plotpos)

    ax.hist(plotdata1, bins=50, alpha=0.4)

    ax.hist(plotdata2, bins=50, alpha=0.4)

    ax.set_xlabel(i)

    

    plotpos = plotpos + 1



plt.show()
# オブジェクトカラムを抽出 (object)

list_cols_cat = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'object':

        list_cols_cat.append(i)

        

print(list_cols_cat)
# 各統計量を作成

# X_train

statics_X_train_cat = pd.DataFrame([df_X_train[list_cols_cat].nunique().values.tolist(),  # ユニーク数

                                df_X_train[list_cols_cat].isnull().sum().values.tolist()],  # 欠損数

                              index=['unique', 'null'],

                              columns=list_cols_cat).T

# X_test

statics_X_test_cat =  pd.DataFrame([df_X_test[list_cols_cat].nunique().values.tolist(),  # ユニーク数

                                df_X_test[list_cols_cat].isnull().sum().values.tolist()],  # 欠損数

                                index=['unique', 'null'],

                                columns=list_cols_cat).T

# y_traine

statics_y_train_cat = pd.DataFrame([df_y_train['loan_condition'].astype('float64').nunique(),  # ユニーク数

                                df_y_train['loan_condition'].astype('float64').isnull().sum()],  # 欠損数

                                index=['unique', 'null'],

                                columns=['loan_condition']).T
statics_X_train_cat
statics_X_test_cat
statics_y_train_cat
# カテゴリ変数として扱うカラムのカラム名をリスト化

# カテゴリ変数として扱うカラムのカラム名をリスト化

list_cols_cat = ['home_ownership',

               'issue_d',

               'purpose',

               'title',

               'zip_code',

               'delinq_2yrs',

               'earliest_cr_line',

               'initial_list_status',

               'application_type',

               'addr_state']



# エンコーダを新たに作成

ce_oe = ce.OrdinalEncoder(cols=list_cols_cat, handle_unknown='impute')



# カテゴリ変数をOrdinal Encodingし、新たなデータフレームを作成

df_X_train_prep = ce_oe.fit_transform(df_X_train) # df_X_trainをエンコード

df_X_test_prep = ce_oe.fit_transform(df_X_test) # df_X_testをエンコード
# grade, sub_gradeは序列があるため、その順序に応じたエンコーディングを行う。

# マップの作成

grade_mapping = {'A': 1,

                 'B': 2,

                 'C': 3,

                 'D': 4,

                 'E': 5,

                 'F': 6,

                 'G': 7

                }



sub_grade_mapping = {'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 

                     'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10, 

                     'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15, 

                     'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 10, 

                     'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25, 

                     'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 20, 

                     'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35

                    }



df_X_train_prep['grade'] = df_X_train_prep['grade'].map(grade_mapping)

df_X_train_prep['sub_grade'] = df_X_train_prep['sub_grade'].map(sub_grade_mapping)



df_X_test_prep['grade'] = df_X_test_prep['grade'].map(grade_mapping)

df_X_test_prep['sub_grade'] = df_X_test_prep['sub_grade'].map(sub_grade_mapping)
# emp_lengthをtrainデータとtestデータから取り出し

list_emp_length_train = df_X_train_prep.emp_length.tolist()

list_emp_length_test = df_X_test_prep.emp_length.tolist()
# <1yearを0yearに置き換える



# リスト作成

for i in range(len(list_emp_length_train)):

    if list_emp_length_train[i] == "< 1 year":

        list_emp_length_train[i] = "0 year"

        

for i in range(len(list_emp_length_test)):

    if list_emp_length_test[i] == "< 1 year":

        list_emp_length_test[i] = "0 year"
# 箱を用意

list_emp_length_train_num = []

list_emp_length_test_num = []



# 数値にマッチするパターン（0～9の文字(数字)の繰り返し)を定義

# pattern = r'([0-9]*)' # これだと< 1yearsで1がうまく抽出されず""となってしまった
# trainデータの処理

for i in range(len(list_emp_length_train)):

    if pd.isnull(list_emp_length_train[i]):

        list_emp_length_train_num.append(np.nan)

    else:

        # temp3 = re.match(pattern, list_emp_length_train[i]) # これだと< 1yearsで1がうまく抽出されず""となってしまった

        # temp2.append(temp3.group())

        temp_data1 = re.sub(r'\D', '', list_emp_length_train[i])

        list_emp_length_train_num.append(temp_data1)

        

# testデータの処理

for i in range(len(list_emp_length_test)):

    if pd.isnull(list_emp_length_test[i]):

        list_emp_length_test_num.append(np.nan)

    else:

        temp_data2 = re.sub(r'\D', '', list_emp_length_test[i])

        list_emp_length_test_num.append(temp_data2)
# 得られた数値のリスト(オブジェクト型)を代入

df_X_train_prep.emp_length = list_emp_length_train_num

df_X_test_prep.emp_length = list_emp_length_test_num

# オブジェクト型を数値型に変換

df_X_train_prep.emp_length = df_X_train_prep.emp_length.astype('float64')

df_X_test_prep.emp_length = df_X_test_prep.emp_length.astype('float64')
# カラムにおける欠損値を、それぞれの中央値で穴埋めする

df_X_train_prep.fillna(df_X_train_prep.median(), inplace=True)

df_y_train.fillna(df_y_train.median(), inplace=True)

df_X_test_prep.fillna(df_X_test_prep.median(), inplace=True)
# 金額系を対数変換

list_cols_num_money = ['installment',

               'loan_amnt',

               'annual_inc',

               'tot_coll_amt',

               'tot_cur_bal']



# 学習データに基づいて複数カラムの標準化を定義

X_train_money_logarithm = np.log1p(df_X_train_prep[list_cols_num_money])

X_test_money_logarithm = np.log1p(df_X_test_prep[list_cols_num_money])



# 変換後のデータで各列を置換

df_X_train_prep[list_cols_num_money] = X_train_money_logarithm

df_X_test_prep[list_cols_num_money] = X_test_money_logarithm
# 数値のカラムを指定

list_cols_num = ['installment',

               'loan_amnt',

               'annual_inc',

               'tot_coll_amt',

               'tot_cur_bal',

               'dti',

               'inq_last_6mths',

               'mths_since_last_delinq',

               'mths_since_last_record',

               'open_acc',

               'pub_rec',

               'revol_bal',

               'revol_util',

               'total_acc',

               'collections_12_mths_ex_med',

               'mths_since_last_major_derog',

               'acc_now_delinq']



# 学習データに基づいて複数カラムの標準化を定義

scaler = StandardScaler()

scaler.fit(df_X_train_prep[list_cols_num])



# 変換後のデータで各列を置換

df_X_train_prep[list_cols_num] = scaler.transform(df_X_train_prep[list_cols_num])

df_X_test_prep[list_cols_num] = scaler.transform(df_X_test_prep[list_cols_num])
# テキストカラムをリストで抜き出し

X_train_text = df_X_train_prep.emp_title

X_test_text = df_X_test_prep.emp_title



# 欠損値をNaNという言葉で埋める

X_train_text.fillna('NaN', inplace=True)

X_test_text.fillna('NaN', inplace=True)
# 全てのemp_titleをTfIdfでベクトル化

vec_all = TfidfVectorizer(max_features=100000)
# emp_titleは全て使う

emp_title_all = pd.concat([X_train_text, X_test_text])
# 全てのemp_titleをTfIdfでベクトル化

vec_all.fit_transform(emp_title_all)
# X_train_text用ベクタライザーの指定

# 辞書はvec_allで抽出したものを使う。

vec_train = TfidfVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)
# X_train_textをベクトル化

X_train_text_tfidf = vec_train.fit_transform(X_train_text)
# X_test_text用ベクタライザーの指定

# 辞書は辞書はvec_allで抽出したものを使う。

vec_test = TfidfVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)
# X_test_textをベクトル化

X_test_text_tfidf = vec_test.fit_transform(X_test_text)
X_test_text_tfidf
# emp_titleをデータフレームからドロップ

df_X_train_prep.drop(['emp_title'], axis=1, inplace=True)

df_X_test_prep.drop(['emp_title'], axis=1, inplace=True)
# Xを指定

# tfidfでベクトル化したテキストカラムをconcatenate

# X = df_X_train_prep.values

# X = np.concatenate((df_X_train_prep.values, X_train_text_tfidf.toarray()), axis=1)



# スパース行列を作成

X_train_prep_sparse = sparse.csr_matrix(df_X_train_prep.values) # TfIdf以外のdenseをsparseに

X = sparse.hstack([X_train_prep_sparse, X_train_text_tfidf])

# 行方向に圧縮

X = X.tocsr()



# yを指定

y = df_y_train.loan_condition.values



# アルゴリズムを指定

# classifier = GradientBoostingClassifier()

# classifier = LogisticRegression()

classifier = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                           importance_type='split', learning_rate=0.05, max_depth=-1,

                           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                           n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                           random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                           subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



# 層化抽出法における分割数を指定(2分割)

skf = StratifiedKFold(n_splits=2, random_state=71, shuffle=True)
# 2回のValidation Scoreを格納するリストを作成

list_cv_auc_score = []

# best iteration回数を格納するリストを作成

list_num_best_iteration = []



# valid_ixを格納するリストを作成

list_valid_ix = []

# pred_validを格納するリストを作成

list_pred_valid = []



d = datetime.datetime.today()

print('start:', d)



t = 0



# 2分割層化抽出によるクロスバリデーションを実行

# 各回のスコアを記録

for train_ix, valid_ix in skf.split(X, y):

    X_train, y_train = X[train_ix], y[train_ix]

    X_valid, y_valid = X[valid_ix], y[valid_ix]

    

    # フィッティング

    # fit_train = classifier.fit(X_train, y_train)

    # LGBM

    fit_train = classifier.fit(X_train, y_train,

                               early_stopping_rounds=200,

                               eval_metric='auc',

                               eval_set=[(X_valid, y_valid)],

                               verbose=100)

    # 予測を実施

    pred_valid = fit_train.predict_proba(X_valid)

    # クラス1に所属する確率を用いてAUCスコアを算出

    v_auc_score = roc_auc_score(y_valid, pred_valid[:, 1])  # 0カラム目はクラス0に分類される確率。

    # AUCスコアを記録

    list_cv_auc_score.append(v_auc_score)

    

    # best_iteraion回数を記録

    num_best_iteration = fit_train.best_iteration_

    list_num_best_iteration.append(num_best_iteration)

    

    # valid_ixを記録

    list_valid_ix = list_valid_ix + valid_ix.tolist()

    

    # pred_validを記録

    list_pred_valid = list_pred_valid + pred_valid[:, 1].tolist()

    

    # タイムスタンプをprint

    t = t + 1

    d = datetime.datetime.today()

    print(t, '_finished:', d)

    

    # スコア表示

    print('AUCは', v_auc_score)

    print('Best Iteration回数は', num_best_iteration)
# validインデックスと、そのインデックスにおける予測結果をDataFrameに

df_pred_valid = pd.DataFrame([list_valid_ix, list_pred_valid]).T
# カラム名設定

df_pred_valid.columns = ['valid_ix', 'pred_valid']

# valid_ixで並び替え

df_pred_valid.sort_values('valid_ix', ascending=True, inplace=True)

# valid_ixをDataFrameのインデックスに

df_pred_valid = df_pred_valid.set_index('valid_ix')
df_X_train['emp_title'].head()
# カテゴリ変数として扱うカラムのカラム名をリスト化

list_cols_cat = ['emp_title']



# エンコーダを作成

ce_oe = ce.OrdinalEncoder(cols=list_cols_cat, handle_unknown='impute')



# カテゴリ変数をOrdinal Encodingし、新たなデータフレームを作成

df_X_train_emp_title_oe = ce_oe.fit_transform(df_X_train)['emp_title'] # df_X_trainをエンコード

df_X_test_emp_title_oe = ce_oe.fit_transform(df_X_test)['emp_title'] # df_X_testをエンコード
df_X_train['issue_d'].values
# 日付をいい感じにやってくれるparseをインポート

from dateutil.parser import parse
# issue_dをdatetime化 (trainデータ)

list_X_train_issue_d = []

[list_X_train_issue_d.append(parse(df_X_train['issue_d'].iloc[i])) for i in range(len(df_X_train['issue_d']))]

list_X_train_issue_d[0:5]
# issue_dをdatetime化 (testデータ)

list_X_test_issue_d = []

[list_X_test_issue_d.append(parse(df_X_test['issue_d'].iloc[j])) for j in range(len(df_X_test['issue_d']))]

list_X_test_issue_d[0:5]
# データをプロット

fig = plt.figure(figsize = (30, 20))



ax1 = fig.add_subplot(2, 2, 1)

ax1.scatter(list_X_train_issue_d, df_X_train_emp_title_oe.values)

ax1.set_ylabel('emp_title(OrdEnc)')

ax1.set_title('emp_title(OrdEnc)(Train Data)')

ax2 = fig.add_subplot(2, 2, 2)

ax2.scatter(list_X_test_issue_d, df_X_test_emp_title_oe.values)

ax2.set_ylabel('emp_title(OrdEnc)(Test Data)')

ax2.set_title('Test Data')

ax3 = fig.add_subplot(2, 2, 3)

ax3.scatter(list_X_train_issue_d, df_pred_valid['pred_valid'])

ax3.set_ylabel('Probability of bad loan')

ax3.set_title('Probability of bad loan (Validation Result)')