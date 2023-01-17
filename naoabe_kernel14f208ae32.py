import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import TimeSeriesSplit



from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier



from hyperopt import fmin, tpe, hp, rand, Trials



import gc
pd.set_option("display.max_columns", None)
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)

df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
#あとの時系列データ分割用にソート

df_train.sort_values('issue_d', inplace=True)
# DataFrameのshapeで行数と列数を確認してみましょう。

df_train.shape, df_test.shape
# 先頭5行をみてみます。

df_train.head()
df_test.head()
df_train[df_train.loan_condition==1].loan_amnt.mean() # 貸し倒れたローンの平均額
# 上の貸し倒れたローンに対するものを参考に、貸し倒れていないローンの平均額を算出みてください。

df_train[df_train.loan_condition==0].loan_amnt.mean()
df_train.describe()
df_test.describe()
f = 'loan_amnt'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

# testデータに対する可視化を記入してみましょう

df_test[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
f = 'purpose'



# value_countsを用いてtrainのpurposeに対して集計結果をみてみましょう。

df_train[f].value_counts() / len(df_train)
# 同様にtestデータに対して

df_test[f].value_counts() / len(df_test)
# 外部データ(statelatlong)を追加

df_statelatlong = pd.read_csv('../input/homework-for-students2/statelatlong.csv')

df_statelatlong
df_train = pd.merge(df_train, df_statelatlong, left_on='addr_state', right_on='State', how='left')

df_test = pd.merge(df_test, df_statelatlong, left_on='addr_state', right_on='State', how='left')

df_train.drop(['State', 'City'], axis=1, inplace=True)

df_test.drop(['State', 'City'], axis=1, inplace=True)
df_train
# 外部データ(SPI)追加

df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'])

df_spi
df_spi['issue_d_year'] = df_spi.date.dt.year

df_spi['issue_d_month'] = df_spi.date.dt.month
df_temp = df_spi.groupby(['issue_d_year', 'issue_d_month'], as_index=False)['close'].mean()

df_temp.head()
# 「issue_d」から年・月データを生成



df_train['issue_d_year'] = df_train.issue_d.dt.year

df_test['issue_d_year'] = df_test.issue_d.dt.year



df_train['issue_d_month'] = df_train.issue_d.dt.month

df_test['issue_d_month'] = df_test.issue_d.dt.month
df_train = df_train.merge(df_temp, on=['issue_d_year', 'issue_d_month'], how='left')

df_test = df_test.merge(df_temp, on=['issue_d_year', 'issue_d_month'], how='left')
df_train
df_test
# 外部データ(US_GDP_by_State)追加

df_US_GDP_by_State = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')

df_US_GDP_by_State
#df_train = pd.merge(df_train, df_US_GDP_by_State, left_on=['City', 'issue_d_year'], right_on=['State', 'year'], how='left')

#df_test = pd.merge(df_test, df_US_GDP_by_State, left_on=['City', 'issue_d_year'], right_on=['State', 'year'], how='left')

#df_train
#plt.figure(figsize=[7,7])

#plt.hist(df_train['State & Local Spending'], density=True, bins=20)

#plt.figure(figsize=[7,7])

#plt.hist(df_train['Gross State Product'], density=True, bins=20)

#plt.figure(figsize=[7,7])

#plt.hist(df_train['Real State Growth %'], density=True, bins=20)

#plt.figure(figsize=[7,7])

#plt.hist(df_train['Population (million)'], density=True, bins=20)
## earliest_cr_lineは、月部分を追加する



#df_train['earliest_cr_line_year'] = df_train.earliest_cr_line.dt.year

#df_test['earliest_cr_line_year'] = df_test.earliest_cr_line.dt.year

df_train['earliest_cr_line_month'] = df_train.earliest_cr_line.dt.month

df_test['earliest_cr_line_month'] = df_test.earliest_cr_line.dt.month
df_train.columns
# 「earliest_cr_line」と「issue_d」の月数差を取る。

df_train['issue_d_earliest_cr_line_diff'] = (df_train.issue_d.dt.year * 12 + df_train.issue_d.dt.month) - (df_train.earliest_cr_line.dt.year * 12 + df_train.earliest_cr_line.dt.month)

df_test['issue_d_earliest_cr_line_diff'] = (df_test.issue_d.dt.year * 12 + df_test.issue_d.dt.month) - (df_test.earliest_cr_line.dt.year * 12 + df_test.earliest_cr_line.dt.month)





## 過去の全アカウント数のうち現行滞納アカウント数の比率

#df_train['acc_now_delinq_total_acc_retio'] = df_train.acc_now_delinq / df_train.total_acc * 100

#df_test['acc_now_delinq_total_acc_retio'] = df_test.acc_now_delinq / df_test.total_acc * 100



## 開いているアカウント数のうち現行滞納アカウント数の比率

#df_train['acc_now_delinq_open_acc_retio'] = df_train.acc_now_delinq / df_train.open_acc * 100

#df_test['acc_now_delinq_open_acc_retio'] = df_test.acc_now_delinq / df_test.open_acc * 100



# 全口座残高のうちのローン額の比率

#df_train['loan_amnt_tot_cur_bal_ratio'] = df_train.loan_amnt / df_train.tot_cur_bal * 100

#df_test['loan_amnt_tot_cur_bal_ratio'] = df_test.loan_amnt / df_test.tot_cur_bal * 100



## 全口座残高のうちの未払いの総回収額の比率

#df_train['tot_coll_amt_tot_cur_bal_ratio'] = df_train.tot_coll_amt / df_train.tot_cur_bal * 100

#df_test['tot_coll_amt_tot_cur_bal_ratio'] = df_test.tot_coll_amt / df_test.tot_cur_bal * 100







# 1%点以下の値は1%点に、99%点以上の値は99%点にclippinngして対数変換



#num_cols_target_cl_lg = ['acc_now_delinq_total_acc_retio', 'acc_now_delinq_open_acc_retio', 'loan_amnt_tot_cur_bal_ratio', 'tot_coll_amt_tot_cur_bal_ratio']

#num_cols_target_cl_lg = ['loan_amnt_tot_cur_bal_ratio']

#num_cols_target_cl_lg = ['loan_amnt_tot_cur_bal_ratio', 'tot_coll_amt_tot_cur_bal_ratio']



#p01_cl_lg = df_train[num_cols_target_cl_lg].quantile(0.01)

#p99_cl_lg = df_train[num_cols_target_cl_lg].quantile(0.99)



#plt.figure(figsize=[7,7])

#df_train[num_cols_target_cl_lg].clip(p01_cl_lg, p99_cl_lg, axis=1).apply(np.log1p).hist(bins=20)

#df_test[num_cols_target_cl_lg].clip(p01_cl_lg, p99_cl_lg, axis=1).apply(np.log1p).hist(bins=20)

#df_train[num_cols_target_cl_lg].apply(np.log1p).hist(bins=20)

#df_test[num_cols_target_cl_lg].apply(np.log1p).hist(bins=20)
# NULLフラグを生成



#df_train['emp_title_null'] = df_train['emp_title'].apply(lambda x : 1 if not x else 0)

#df_train['emp_length_null'] = df_train['emp_length'].apply(lambda x : 1 if x == 'n/a' else 0)

#df_train['annual_inc_null'] = df_train['annual_inc'].apply(lambda x : 1 if not x else 0)

#df_train['title_null'] = df_train['title'].apply(lambda x : 1 if not x else 0)

#df_train['dti_null'] = df_train['dti'].apply(lambda x : 1 if not x else 0)

#df_train['delinq_2yrs_null'] = df_train['delinq_2yrs'].apply(lambda x : 1 if not x else 0)

#df_train['earliest_cr_line_null'] = df_train['earliest_cr_line'].apply(lambda x : 1 if not x else 0)

#df_train['inq_last_6mths_null'] = df_train['inq_last_6mths'].apply(lambda x : 1 if not x else 0)

#df_train['mths_since_last_delinq_null'] = df_train['mths_since_last_delinq'].apply(lambda x : 1 if not x else 0)

#df_train['mths_since_last_record_null'] = df_train['mths_since_last_record'].apply(lambda x : 1 if not x else 0)

#df_train['open_acc_null'] = df_train['open_acc'].apply(lambda x : 1 if not x else 0)

#df_train['pub_rec_null'] = df_train['pub_rec'].apply(lambda x : 1 if not x else 0)

#df_train['revol_util_null'] = df_train['revol_util'].apply(lambda x : 1 if not x else 0)

#df_train['total_acc_null'] = df_train['total_acc'].apply(lambda x : 1 if not x else 0)

#df_train['collections_12_mths_ex_med_null'] = df_train['collections_12_mths_ex_med'].apply(lambda x : 1 if not x else 0)

#df_train['mths_since_last_major_derog_null'] = df_train['mths_since_last_major_derog'].apply(lambda x : 1 if not x else 0)

#df_train['acc_now_delinq_null'] = df_train['acc_now_delinq'].apply(lambda x : 1 if not x else 0)

#df_train['tot_coll_amt_null'] = df_train['tot_coll_amt'].apply(lambda x : 1 if not x else 0)

#df_train['tot_cur_bal_null'] = df_train['tot_cur_bal'].apply(lambda x : 1 if not x else 0)



#df_test['emp_title_null'] = df_test['emp_title'].apply(lambda x : 1 if not x else 0)

#df_test['emp_length_null'] = df_test['emp_length'].apply(lambda x : 1 if x == 'n/a' else 0)

#df_test['annual_inc_null'] = df_test['annual_inc'].apply(lambda x : 1 if not x else 0)

#df_test['title_null'] = df_test['title'].apply(lambda x : 1 if not x else 0)

#df_test['dti_null'] = df_test['dti'].apply(lambda x : 1 if not x else 0)

#df_test['delinq_2yrs_null'] = df_test['delinq_2yrs'].apply(lambda x : 1 if not x else 0)

#df_test['earliest_cr_line_null'] = df_test['earliest_cr_line'].apply(lambda x : 1 if not x else 0)

#df_test['inq_last_6mths_null'] = df_test['inq_last_6mths'].apply(lambda x : 1 if not x else 0)

#df_test['mths_since_last_delinq_null'] = df_test['mths_since_last_delinq'].apply(lambda x : 1 if not x else 0)

#df_test['mths_since_last_record_null'] = df_test['mths_since_last_record'].apply(lambda x : 1 if not x else 0)

#df_test['open_acc_null'] = df_test['open_acc'].apply(lambda x : 1 if not x else 0)

#df_test['pub_rec_null'] = df_test['pub_rec'].apply(lambda x : 1 if not x else 0)

#df_test['revol_util_null'] = df_test['revol_util'].apply(lambda x : 1 if not x else 0)

#df_test['total_acc_null'] = df_test['total_acc'].apply(lambda x : 1 if not x else 0)

#df_test['collections_12_mths_ex_med_null'] = df_test['collections_12_mths_ex_med'].apply(lambda x : 1 if not x else 0)

#df_test['mths_since_last_major_derog_null'] = df_test['mths_since_last_major_derog'].apply(lambda x : 1 if not x else 0)

#df_test['acc_now_delinq_null'] = df_test['acc_now_delinq'].apply(lambda x : 1 if not x else 0)

#df_test['tot_coll_amt_null'] = df_test['tot_coll_amt'].apply(lambda x : 1 if not x else 0)

#df_test['tot_cur_bal_null'] = df_test['tot_cur_bal'].apply(lambda x : 1 if not x else 0)
df_train.columns
# 不要な特徴量を削除

#df_train = df_train.drop(['earliest_cr_line', 'issue_d', 'State_x', 'City', 'issue_d_year', 'State_y', 'year'], axis=1)

#df_test = df_test.drop(['earliest_cr_line', 'issue_d', 'State_x', 'City', 'issue_d_year', 'State_y', 'year'], axis=1)

#df_train = df_train.drop(['earliest_cr_line', 'issue_d', 'State', 'City'], axis=1)

#df_test = df_test.drop(['earliest_cr_line', 'issue_d', 'State', 'City'], axis=1)

df_train = df_train.drop(['earliest_cr_line', 'issue_d'], axis=1)

df_test = df_test.drop(['earliest_cr_line', 'issue_d'], axis=1)
#Xとyに分割



y_train = df_train['loan_condition']

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
# clippingしてみる

# 列ごとに学習データの1%点、99%点を計算

num_cols = ['annual_inc', 'dti', 'revol_bal', 'tot_coll_amt', 'tot_cur_bal']



p01 = X_train[num_cols].quantile(0.01)

p99 = X_train[num_cols].quantile(0.99)



# 1%点以下の値は1%点に、99%点以上の値は99%点にclippinngする

plt.figure(figsize=[7,7])

X_train[num_cols].hist(bins=20)

X_test[num_cols].hist(bins=20)

X_train[num_cols].clip(p01, p99, axis=1).hist(bins=20)

X_test[num_cols].clip(p01, p99, axis=1).hist(bins=20)

X_train[num_cols].clip(p01, p99, axis=1).apply(np.log1p).hist(bins=20)

X_test[num_cols].clip(p01, p99, axis=1).apply(np.log1p).hist(bins=20)
num_cols_target_cl = ['dti']

num_cols_target_cl_lg = ['annual_inc', 'revol_bal', 'tot_cur_bal']



p01_cl = X_train[num_cols_target_cl].quantile(0.01)

p99_cl = X_train[num_cols_target_cl].quantile(0.99)

p01_cl_lg = X_train[num_cols_target_cl_lg].quantile(0.01)

p99_cl_lg = X_train[num_cols_target_cl_lg].quantile(0.99)



# 1%点以下の値は1%点に、99%点以上の値は99%点にclippinngする

plt.figure(figsize=[7,7])

X_train[num_cols_target_cl].clip(p01_cl, p99_cl, axis=1).hist(bins=20)

X_test[num_cols_target_cl].clip(p01_cl, p99_cl, axis=1).hist(bins=20)

X_train[num_cols_target_cl_lg].clip(p01_cl_lg, p99_cl_lg, axis=1).apply(np.log1p).hist(bins=20)

X_test[num_cols_target_cl_lg].clip(p01_cl_lg, p99_cl_lg, axis=1).apply(np.log1p).hist(bins=20)



#X_train[num_cols] = X_train[num_cols].clip(p01, p99, axis=1)

#X_test[num_cols] = X_test[num_cols].clip(p01, p99, axis=1)
# tot_coll_amtはYeo-Johnson変換



#from sklearn.preprocessing import PowerTransformer



#col = ['tot_coll_amt']



# 学習データに基づいて複数カラムのYeo-Johnson変換を定義

#pt = PowerTransformer(method='yeo-johnson')

#pt.fit(X_train[col])



# 変換後のデータで各列を置換

#X_train_yj_temp= pt.transform(X_train[col])

#X_test_yj_temp = pt.transform(X_test[col])

#X_train_yj_temp
#X_train['tot_coll_amt'] = X_train_yj_temp

#X_test['tot_coll_amt'] = X_test_yj_temp

#plt.figure(figsize=[7,7])

#X_train['tot_coll_amt'].hist(bins=20)
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
X_train['emp_title'].head(10) # カテゴリよりテキストとして扱ったほうが良いかもしれない
# titleはテキストとして扱うか？

X_train.groupby(['purpose', 'title'], as_index=False).size()

#sub_grade_values = X_train['sub_grade'].drop_duplicates().sort_values()

#sub_grade_label = list(range(1, len(sub_grade_values) + 1))



#sub_grade_dict = {}

#sub_grade_dict.update(zip(sub_grade_values, sub_grade_label))

#sub_grade_dict
target = 'loan_condition'

t_encoding_col = ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 'earliest_cr_line_month', 'issue_d_month']

#t_encoding_col = ['emp_length', 'home_ownership', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 'earliest_cr_line_month', 'issue_d_month']



for i, t_col in enumerate(t_encoding_col):



    X_temp = pd.concat([X_train, y_train], axis=1)



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([t_col])[target].mean()

    enc_test = X_test[t_col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([t_col])[target].mean()

        enc_train.iloc[val_ix] = X_val[t_col].map(summary)

        

    X_train[t_col] = enc_train

    X_test[t_col] = enc_test
X_train['zip_code']
enc_test
cats
TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



cats.remove('emp_title')
TXT_train
X_train.head()
X_test.head()
#Ordinalエンコーディング対象外の特徴量を除外する

# ※最終的にtitle、zip_codeのみ



cats.remove('grade')

cats.remove('sub_grade')

cats.remove('emp_length')

cats.remove('home_ownership')

cats.remove('purpose')

cats.remove('addr_state')

cats.remove('initial_list_status')

cats.remove('application_type')

#cats.remove('zip_code')

cats
encoder = OrdinalEncoder(cols=cats)

encoder
X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])
X_train.head()
X_test.head()
## emp_lengthを順序尺度のエンコーディング

#emp_length_mapping = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': np.nan}

#X_train['emp_length'] = X_train['emp_length'].map(emp_length_mapping)

#X_test['emp_length'] = X_test['emp_length'].map(emp_length_mapping)

#X_train['emp_length']
## gradeを順序尺度のエンコーディング

#grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

#X_train['grade'] = X_train['grade'].map(grade_mapping)

#X_test['grade'] = X_test['grade'].map(grade_mapping)

#X_train['grade']
## sub_gradeを順序尺度のエンコーディング



##sub_grade_mapping = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G':'7'}



#sub_grade_values = X_train['sub_grade'].drop_duplicates().sort_values()

#sub_grade_label = list(range(1, len(sub_grade_values) + 1))



#sub_grade_mapping = {}

#sub_grade_mapping.update(zip(sub_grade_values, sub_grade_label))



#X_train['sub_grade'] = X_train['sub_grade'].map(sub_grade_mapping)

#X_test['sub_grade'] = X_test['sub_grade'].map(sub_grade_mapping)
# 以下を参考に自分で書いてみましょう 

X_train.drop(['emp_title'], axis=1, inplace=True)

X_test.drop(['emp_title'], axis=1, inplace=True)



#X_train.fillna(X_train.median(), inplace=True)

#X_test.fillna(X_train.median(), inplace=True)



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
# NLTKから英語のstop wordsを読み込んでリストから除外する

#import nltk

from nltk.corpus import stopwords

stops = stopwords.words("english")
TXT_train = TXT_train.fillna('NULL')

TXT_test = TXT_test.fillna('NULL')
TXT_train_array = []

for col in TXT_train.values:

    TXT_train_array.append(col)



TXT_test_array = []

for col in TXT_test.values:

    TXT_test_array.append(col)
from nltk.stem.porter import PorterStemmer as PS

ps = PS()

words_stem_train = [' '.join([ps.stem(w) for w in s.split(' ')]) for s in TXT_train_array]

words_stem_test = [' '.join([ps.stem(w) for w in s.split(' ')]) for s in TXT_test_array]
col_name =['emp_title']

TXT_train = pd.DataFrame(words_stem_train, columns=col_name)

TXT_test = pd.DataFrame(words_stem_test, columns=col_name)
#tfidf = TfidfVectorizer(max_features=1000, use_idf=True)

tfidf = TfidfVectorizer(max_features=100, stop_words=stops)

TXT_train_2 = tfidf.fit_transform(TXT_train.emp_title)

TXT_test_2 = tfidf.transform(TXT_test.emp_title)
del TXT_train, TXT_test

gc.collect()
TXT_train_2
TXT_train_3 = pd.DataFrame(TXT_train_2.toarray(), columns=tfidf.get_feature_names())

TXT_test_3 = pd.DataFrame(TXT_test_2.toarray(), columns=tfidf.get_feature_names())
del TXT_train_2, TXT_test_2

gc.collect()
TXT_train_3
TXT_train_3['key'] = list(range(len(TXT_train_3.index)))

TXT_test_3['key'] = list(range(len(TXT_test_3.index)))

X_train['key'] = list(range(len(X_train.index)))

X_test['key'] = list(range(len(X_test.index)))
TXT_train_3
X_train = pd.merge(X_train, TXT_train_3, how='left', on='key')

X_test = pd.merge(X_test, TXT_test_3, how='left', on='key')

del TXT_train_3, TXT_test_3

gc.collect()



X_train.drop(['key'], axis=1, inplace=True)

X_test.drop(['key'], axis=1, inplace=True)
#X_train.drop(['title'], axis=1, inplace=True)

#X_test.drop(['title'], axis=1, inplace=True)

X_train.drop(['title', 'zip_code'], axis=1, inplace=True)

X_test.drop(['title', 'zip_code'], axis=1, inplace=True)
scores = []



tss = TimeSeriesSplit(n_splits=3)



for i, (train_ix, test_ix) in tqdm(enumerate(tss.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier()

    clf = LGBMClassifier()

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))

    

print(np.mean(scores))

print(scores)
X_train_sp = sp.sparse.csr_matrix(X_train.values)

X_test_sp = sp.sparse.csr_matrix(X_test.values)

X_train.drop(['issue_d_year'], axis=1, inplace=True)

X_test.drop(['issue_d_year'], axis=1, inplace=True)
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier()

    clf = LGBMClassifier()

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1]
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
submission.head()
# Feature Importance

fti = clf.feature_importances_   



print('Feature Importances:')

for i, feat in enumerate(X_train):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=30, ax=ax, importance_type='gain')