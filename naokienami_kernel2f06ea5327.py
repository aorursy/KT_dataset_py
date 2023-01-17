# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss, roc_curve, confusion_matrix, plot_roc_curve

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier



import seaborn as sns



from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
# input_dir = '../input/homework-for-students4/'

input_dir = '../input/homework-for-students4plus/'



# 表示可能範囲を増やしておく

pd.set_option("display.max_rows", 100)

pd.set_option("display.max_columns", 50)
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv(input_dir + 'train.csv', index_col=0)#, skiprows=lambda x: x%2000!=0)

df_test = pd.read_csv(input_dir + 'test.csv', index_col=0)#, skiprows=lambda x: x%2000!=0)



# 緯度経度情報をくっつけておく

lat_long = pd.read_csv(input_dir + 'statelatlong.csv').drop('City', axis=1)

df_train = df_train.merge(lat_long, how='left', left_on='addr_state', right_on='State').drop('State', axis=1)

df_test = df_test.merge(lat_long, how='left', left_on='addr_state', right_on='State').drop('State', axis=1)



# トレーニングデータの先頭5行をみてみます。

df_train.head()
print(df_train.nunique())
# テキスト特徴量と結果に順序性がありそうかを確認する

# grade, sub_gradeとloan_condition

tmp1 = df_train[['grade', 'sub_grade', 'loan_condition']]

tmp1_true = tmp1[tmp1['loan_condition'] == 1]

tmp_1_true_gb = tmp1_true.groupby(['grade', 'sub_grade']).count()

tmp_1_true_gb.sort_values('loan_condition', ascending=False).head(100)
tmp1_false = tmp1[tmp1['loan_condition'] == 0]

tmp_1_false_gb = tmp1_false.groupby(['grade', 'sub_grade']).count()

tmp_1_false_gb.sort_values('loan_condition', ascending=False).head(100)



# 返せている人の割合でみる

print(tmp_1_true_gb/ (tmp_1_true_gb+tmp_1_false_gb))
# テキスト特徴量と結果に順序性がありそうかを確認する

# emp_lengthとloan_condition

tmp2 = df_train[['emp_length', 'loan_condition']]

tmp2_true = tmp2[tmp1['loan_condition'] == 1]

tmp_2_true_gb = tmp2_true.groupby('emp_length').count()

tmp_2_true_gb.sort_values('loan_condition', ascending=False).head(100)
tmp2 = df_train[['emp_length', 'loan_condition']]

tmp2_false = tmp2[tmp1['loan_condition'] == 0]

tmp_2_false_gb = tmp2_false.groupby('emp_length').count()

tmp_2_false_gb.sort_values('loan_condition', ascending=False).head(100)



# 返せている人の割合でみる

print(tmp_2_true_gb/ (tmp_2_true_gb+tmp_2_false_gb))
# テストデータの先頭5行をみてみます。

df_test.head()
print(df_test.nunique())
# トレーニングデータの中身について確認

df_train.info()
# テストデータの中身について確認

df_test.info()
# トレーニングデータとテストデータの数値データの傾向を可視化して確認してみる

# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

# まずは数値特徴量だけを取り出してみます。

X_train_num = df_train.drop(cats, axis=1).fillna(-9999)

X_test_num = df_test.drop(cats, axis=1).fillna(-9999)

        

for f in X_test_num.columns:

    plt.figure(figsize=[7,7])

    X_train_num[f].hist(density=True, alpha=0.5, bins=20)

    X_test_num[f].hist(density=True, alpha=0.5, bins=20)

    plt.xlabel(f)

    plt.ylabel('density')

    plt.show()
# 数値データの相関行列を可視化してみる

import seaborn as sns

# N 行 M 列を M 行 N 列に変換して相関行列を計算する

correlation_matrix = np.corrcoef(X_train_num.transpose())





plt.figure(figsize=(15, 15)) 

# 相関行列のヒートマップを描く

sns.heatmap(correlation_matrix, annot=True,

            xticklabels=X_train_num.columns,

            yticklabels=X_train_num.columns)



# グラフを表示する

plt.show()



# installment, annual_inc, revol_bal, tot_cur_balあたりを中心に特徴量エンジニアリングを行ってみる
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test.copy()
# 特徴量グループごとにリスト化

cat = []

num = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        if col != 'emp_title':

            cat.append(col)

    else:

        if col != 'issue_d':

            num.append(col)



# train/test

# 特徴量タイプごとに分割する

cat_train = X_train[cat]#.drop('issue_d', axis=1)

txt_train = X_train.emp_title

num_train = X_train[num]

time_train = pd.DataFrame(pd.to_datetime(X_train['issue_d']))



cat_test = X_test[cat]#.drop('issue_d', axis=1)

txt_test = X_test.emp_title

num_test = X_test[num]

time_test = pd.DataFrame(pd.to_datetime(X_test['issue_d']))
# # text2feature ※時間があれば実施

# from sklearn.feature_extraction.text import TfidfVectorizer



# # 欠損値を埋める

# txt_train.replace(np.nan, "", inplace=True)

# txt_test.replace(np.nan, "", inplace=True)



# tfidf_vect = TfidfVectorizer()

# train_tfidf = tfidf_vect.fit_transform(txt_train)

# test_tfidf = tfidf_vect.fit_transform(txt_test)
# cat2feature

# grade, sub_grade, emp_lengthは順序をもった数値に置き換えてみる

dic_grade = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}

dic_sub_grade = {'A1':31, 'B1':26, 'C1':21, 'D1':16, 'E1':11, 'F1':6, 'G1':1,

                 'A2':32, 'B2':27, 'C2':22, 'D2':17, 'E2':12, 'F2':7, 'G2':2,

                 'A3':33, 'B3':28, 'C3':23, 'D3':18, 'E3':13, 'F3':8, 'G3':3,

                 'A4':34, 'B4':29, 'C4':24, 'D4':19, 'E4':14, 'F4':9, 'G4':4,

                 'A5':35, 'B5':30, 'C5':25, 'D5':20, 'E5':15, 'F5':10, 'G5':5}

dic_emp_length = {'< 1 year':1, '2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,

                  '7 years':7,'8 years':8,'9 years':9,'10+ years':10}



cat_train['grade'] = cat_train['grade'].replace(dic_grade)

cat_test['grade'] = cat_test['grade'].replace(dic_grade)

cat_train['sub_grade'] = cat_train['sub_grade'].replace(dic_sub_grade)

cat_test['sub_grade'] = cat_test['sub_grade'].replace(dic_sub_grade)

cat_train['emp_length'] = cat_train['emp_length'].replace(dic_emp_length)

cat_test['emp_length'] = cat_test['emp_length'].replace(dic_emp_length)



# ordinalエンコーディングを行う

oe = OrdinalEncoder(cols=cat[2:])



cat_train = oe.fit_transform(cat_train)

cat_test = oe.transform(cat_test)
# 関係ありそうな特徴量を作ってみる

# Grade × Sub_Grade

cat_train['grade_rank'] = cat_train['grade'] * cat_train['sub_grade']

cat_test['grade_rank'] = cat_test['grade'] * cat_test['sub_grade']

# num2feature

# 対数変換を行う

li = ['loan_amnt','installment','mths_since_last_record','mths_since_last_delinq','open_acc','total_acc']

num_train[li] = num_train[li].apply(np.log1p)

num_test[li] = num_test[li].apply(np.log1p)



# 二乗特徴量を作成してみる

num_train['installment2'] = num_train['installment'] ** 2 

num_train['annual_inc2'] = num_train['annual_inc'] ** 2 

num_train['revol_bal2'] = num_train['revol_bal'] ** 2 

num_train['tot_cur_bal2'] = num_train['tot_cur_bal'] ** 2 

num_test['installment2'] = num_test['installment'] ** 2 

num_test['annual_inc2'] = num_test['annual_inc'] ** 2 

num_test['revol_bal2'] = num_test['revol_bal'] ** 2 

num_test['tot_cur_bal2'] = num_test['tot_cur_bal'] ** 2 

# loan_amountと相関高そうなものの交互作用に期待してみる

# 値が0～1の間の時があると悪さしそうなので一旦保留

# num_train['installment_annual_inc'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['annual_inc'] ** 2)

# num_train['installment_revol_bal'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['revol_bal'] ** 2)

# num_train['installment_tot_cur_bal'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['tot_cur_bal'] ** 2)

# num_train['installment_tot_cur_bal'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['tot_cur_bal'] ** 2)

# num_train['installment_tot_cur_bal'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['tot_cur_bal'] ** 2)

# num_train['installment_tot_cur_bal'] = numpy.sqrt(num_train['installment'] ** 2) * numpy.sqrt(num_train['tot_cur_bal'] ** 2)

# time2feature

import math

split_year = lambda x: int(str(x).split('-')[0])

# 月は循環性を意識してみる

split_month_x = lambda x: math.cos(2.0 * math.pi / (int(str(x).split('-')[1])))

split_month_y = lambda x: math.sin(2.0 * math.pi / (int(str(x).split('-')[1])))



time_train['year'] = time_train['issue_d'].apply(split_year)

time_train['month_x'] = time_train['issue_d'].apply(split_month_x)

time_train['month_y'] = time_train['issue_d'].apply(split_month_y)

# # 2015年以降は重要そうなのでフラグを付与してみる

time_train.loc[time_train['year'] >= 2015, 'year_flag'] = 1

time_train.loc[time_train['year'] < 2015, 'year_flag'] = 0



time_test['year'] = time_test['issue_d'].apply(split_year)

time_test['month_x'] = time_test['issue_d'].apply(split_month_x)

time_test['month_y'] = time_test['issue_d'].apply(split_month_y)

# # 2015年以降は重要そうなのでフラグを付与してみる

time_test.loc[time_test['year'] >= 2015, 'year_flag'] = 1

time_test.loc[time_test['year'] < 2015, 'year_flag'] = 0



time_train.drop('issue_d', inplace=True, axis=1)

time_test.drop('issue_d', inplace=True, axis=1)
# 作成した特徴量セットを連結する

X_train = pd.concat([cat_train, num_train, time_train], axis=1)

X_test = pd.concat([cat_test, num_test, time_test], axis=1)



# yearが2015年以降のデータのみを利用する

# X_train = X_train_tmp.query('year >= 2015')

# X_test = X_test_tmp.query('year >= 2015')



# cat_train = X_train[cat]

# txt_train = X_train.emp_title

# num_train = X_train[num]

# time_train = pd.DataFrame(pd.to_datetime(X_train['issue_d']))
# nan_pattern特徴量を作ってみる

# nanなら1、以外は0を入れる文字列を作成

create_nan_pattern = lambda x: func_create_nan_pattern(x)

def func_create_nan_pattern(row):

    s = ''

    for f in row.index:

        if row[f] != row[f]:

            s = s + '1'

        else:

            s = s + '0'

    return s



# nan列の数をカウント

count_nan_pattern = lambda x: func_count_nan_pattern(x)

def func_count_nan_pattern(row):

    s = 0

    for f in row.index:

        if row[f] != row[f]:

            s = s + 1

    return s



# 行ごとに適用する

X_train['nan_sum'] = X_train.apply(count_nan_pattern, axis=1)

X_train['nan_pattern'] = X_train.apply(create_nan_pattern, axis=1)

X_test['nan_sum'] = X_test.apply(count_nan_pattern, axis=1)

X_test['nan_pattern'] = X_test.apply(create_nan_pattern, axis=1)



# nan_patternにはordinalエンコーディングを行う

oe = OrdinalEncoder(cols='nan_pattern')

X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)

# 欠損値の処理を行う

X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
# # ローカル環境作業用

# # ここまでの結果を出力しておく

# pd.concat([X_train,y_train], axis=1).to_csv(input_dir + 'train_.csv')

# X_test.to_csv(input_dir + 'test_.csv')
# # ローカル環境作業用

# df_train = pd.read_csv(input_dir + 'train_.csv', index_col=0)#, skiprows=lambda x: x%2000!=0)

# df_test = pd.read_csv(input_dir + 'test_.csv', index_col=0)#, skiprows=lambda x: x%2000!=0)



# X_train = df_train.drop('loan_condition', axis=1)

# y_train = df_train.loan_condition



# X_test = df_test.copy()
# scores = []



# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# # GradientBoostingClassifier

# for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

#     X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#     X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

#     clf = GradientBoostingClassifier()

    

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

    

#     print('CV Score of Fold_%d is %f' % (i, score))
# # light GBM

# # 学習用と検証用に分割する

# X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)

# clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

#                                 importance_type='split', learning_rate=0.05, max_depth=-1,

#                                 min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                                 n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                                 random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                                 subsample=1.0, subsample_for_bin=200000, subsample_freq=0)





# clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

# clf.booster_.feature_importance(importance_type='gain')
scores = []

y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix]



    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                    importance_type='split', learning_rate=0.05, max_depth=-1,

                                    min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                    n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                    random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                    subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(y_val, y_pred))



    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく



scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 5 # 最後にfold数で割る
# # サポートベクターマシン（SVM）

# from sklearn.svm import SVC

# from sklearn.ensemble import VotingClassifier

# svm_clf = SVC(probability=True, random_state=71)

# svm_clf.fit(X_train, y_train)

# 時間があれば実施
# 時間があれば実施
# # 全データで再学習し、testに対して予測する

# clf.fit(X_train, y_train)

# y_pred = clf.predict_proba(X_test)[:,1]



# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv(input_dir + 'sample_submission.csv', index_col=0)



submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')