import numpy as np

import scipy as sp

import pandas as pd

import time

import gc

import re

import random

from pandas import DataFrame, Series

import datetime as datetime



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold,ShuffleSplit,KFold,GridSearchCV, RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm.notebook import tqdm

from sklearn.preprocessing import quantile_transform

from sklearn.ensemble import VotingClassifier,RandomForestClassifier

from sklearn.preprocessing import StandardScaler



import lightgbm as lgb

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from hyperopt import fmin, tpe, hp, rand, Trials



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



# 乱数シード固定

seed_everything(2020)
#df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, skiprows=lambda x: x%20!=0)

df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0)

#df_gdp = pd.read_csv('../input/homework-for-students4plus/US_GDP_by_State.csv', index_col=0)

#df_zipcode = pd.read_csv('../input/homework-for-students4plus/free-zipcode-database.csv', index_col=0)

#df_spi = pd.read_csv('../input/homework-for-students4plus/spi.csv', index_col=0)

#df_state = pd.read_csv('../input/homework-for-students4plus/statelatlong.csv', index_col=0)

df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0)
display(df_train.shape)

display(df_test.shape)
# engeenering

df_train['istrain'] = 1

df_test['istrain'] = 0

df_train_test = pd.concat([df_train, df_test], axis=0)
display(df_train_test.shape)
df_train_test
df_train_test['Datetime'] = pd.to_datetime(df_train_test.issue_d,format='%b-%Y')
display(df_train_test.shape)
print('emp_length', df_train_test['emp_length'].sort_values().unique())
df_train_test['grade_rank']=df_train_test['grade']

df_train_test['sub_grade_rank']=df_train_test['sub_grade']

df_train_test['emp_length_rank']=df_train_test['emp_length']

df_train_test=df_train_test.replace({'grade_rank':{'C':3, 'D':4, 'B':2, 'F':6, 'A':1, 'E':5, 'G':7}})





df_train_test=df_train_test.replace({'sub_grade_rank':{'A1':1, 'A2':2, 'A3':3, 'A4':4, 'A5':5, 'B1':6, 'B2':7, 'B3':8, 'B4':9, 'B5':10, 'C1':11, 'C2':12, 'C3':13, 'C4':14,

 'C5':15, 'D1':16, 'D2':17, 'D3':18, 'D4':19, 'D5':20, 'E1':21, 'E2':22, 'E3':23, 'E4':24, 'E5':25, 'F1':26, 'F2':27, 'F3':28,

 'F4':29, 'F5':30, 'G1':31, 'G2':32, 'G3':33, 'G4':34, 'G5':35}})





df_train_test['emp_length_y']=df_train_test['emp_length'].replace('.*(\d+).*', r'\1', regex=True).fillna('-9999').astype('int32')



df_train_test=df_train_test.replace({'emp_length_rank':{'< 1 year':11, '1 year':10, '10+ years':9, '2 years':8, '3 years':7, '4 years':6, '5 years':5, '6 years':4,

 '7 years':3, '8 years':2, '9 years':1}})



df_train_test['grade_rank'] = df_train_test['grade_rank'].astype('object').fillna('#', inplace=True)



df_train_test['sub_grade_rank'] = df_train_test['sub_grade_rank'].astype('object').fillna('#', inplace=True)



df_train_test['emp_length_rank'] = df_train_test['emp_length_rank'].astype('object').fillna('#', inplace=True)

#月

df_train_test['issue_month']=df_train_test['issue_d'].replace('(.*)-\d+', r'\1', regex=True)



# 数値の四則演算

df_train_test['loan_amnt_x_dti']=df_train_test['loan_amnt'] * df_train_test['dti']

df_train_test['installment_dv_dti']=round(df_train_test['installment'] / df_train_test['dti'])

X_train = df_train_test[df_train_test['istrain'] == 1]

X_test  = df_train_test[df_train_test['istrain'] == 0]
display(X_train.shape)

display(X_test.shape)
# 複数の列を削除

#drop_tain_col = ['emp_title','issue_d','zip_code','addr_state','earliest_cr_line','loan_condition']

#drop_test_col = ['emp_title','issue_d','zip_code','addr_state','earliest_cr_line']

#drop_tain_col = ['issue_d','earliest_cr_line','loan_condition']

#drop_test_col = ['issue_d','earliest_cr_line']

drop_tain_col = ['issue_d','loan_condition','Datetime', 'istrain']

drop_test_col = ['issue_d','Datetime', 'istrain','loan_condition']

#drop_tain_col = ['loan_condition','Datetime']

#drop_test_col = ['Datetime']



#drop_tain_col = ['issue_d','loan_condition','year','index','City','Datetime']

#drop_test_col = ['issue_d','year','index','City','Datetime']
X_train['Datetime']
X_train[(X_train['Datetime'] >= datetime.datetime(2014,1,1)) & (X_train['Datetime'] <= datetime.datetime(2015,12,31))]
display(X_train.shape)

display(X_test.shape)
# 不要な列を削除してデータをXYに分割



X_train = X_train[(X_train['Datetime'] >= datetime.datetime(2014,1,1)) & (X_train['Datetime'] <= datetime.datetime(2015,12,31))]

X_train['datediff'] = (pd.to_datetime(X_train['issue_d'], format='%b-%Y') - pd.to_datetime(X_train['earliest_cr_line'], format='%b-%Y')).dt.days



y_train = X_train.loan_condition



X_train = X_train.drop(drop_tain_col, axis=1)


#X_train = df_train.drop(['loan_condition'], axis=1)



X_test['datediff'] = (pd.to_datetime(X_test['issue_d'], format='%b-%Y') - pd.to_datetime(X_test['earliest_cr_line'], format='%b-%Y')).dt.days

X_test = X_test.drop(drop_test_col, axis=1)



gc.collect()
display(X_train.shape)

display(X_test.shape)
# 標準化

num_cols_std = ['loan_amnt','installment','mths_since_last_delinq','mths_since_last_record','open_acc','revol_util',

                'mths_since_last_major_derog']

scaler = StandardScaler()

scaler.fit(X_train[num_cols_std])

# 変換後のデータで各列を置換

X_train[num_cols_std] = scaler.transform(X_train[num_cols_std])

X_test[num_cols_std] = scaler.transform(X_test[num_cols_std])

# とりあえずいろんな列に対数変換してみる。



#X_train['inst_dev_totcurbal']=X_train['installment']/X_train['tot_cur_bal']

#X_train['inst_dev_totcurbal']=X_test['installment']/X_test['tot_cur_bal']

#X_train['annin_dev_loam']=X_train['annual_inc']/X_train['loan_amnt']

#X_test['annin_dev_loam']=X_test['annual_inc']/X_test['loan_amnt']

#X_train['annual_inc_log2']=np.floor(np.log2(X_train['annual_inc']))



#対数変換

#X_train['annual_inc_log1p'] = X_train['annual_inc'].apply(np.log1p)

#X_train['revol_bal_log1p'] = X_train['revol_bal'].apply(np.log1p)

#X_train['tot_cur_bal_log1p'] = X_train['tot_cur_bal'].apply(np.log1p)

#X_test['annual_inc_log1p'] = X_test['annual_inc'].apply(np.log1p)

#X_test['revol_bal_log1p'] = X_test['revol_bal'].apply(np.log1p)

#X_test['tot_cur_bal_log1p'] = X_test['tot_cur_bal'].apply(np.log1p)

#X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

#X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

#X_train['revol_bal'] = X_train['revol_bal'].apply(np.log1p)

#X_train['delinq_2yrs'] = X_train['delinq_2yrs'].apply(np.log1p)



X_train['tot_cur_bal'] = X_train['tot_cur_bal'].apply(np.log1p)



X_train['inq_last_6mths'] = X_train['inq_last_6mths'].apply(np.log1p)



#X_train['close'] = X_train['close'].apply(np.log1p)

#X_train['pub_rec'] = X_train['pub_rec'].apply(np.log1p)

#X_train['pub_rec']=np.floor(np.log2(X_train['pub_rec']+1))

#X_train['pubcollections_12_mths_ex_med_rec'] = X_train['collections_12_mths_ex_med'].apply(np.log1p)

#X_train['acc_now_delinq'] = X_train['acc_now_delinq'].apply(np.log1p)

#X_train['tot_coll_amt'] = X_train['tot_coll_amt'].apply(np.log1p)

#X_train['collections_12_mths_ex_med']=np.floor(np.log2(X_train['collections_12_mths_ex_med']+1))

#X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)

#X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)

#X_test['revol_bal'] = X_test['revol_bal'].apply(np.log1p)



X_test['tot_cur_bal'] = X_test['tot_cur_bal'].apply(np.log1p)



X_test['inq_last_6mths'] = X_test['inq_last_6mths'].apply(np.log1p)



#X_test['delinq_2yrs'] = X_test['delinq_2yrs'].apply(np.log1p)

#X_test['close'] = X_test['close'].apply(np.log1p)

#X_test['pub_rec'] = X_test['pub_rec'].apply(np.log1p)

#X_test['pub_rec']=np.floor(np.log2(X_test['pub_rec']+1))

#X_test['collections_12_mths_ex_med'] = X_test['collections_12_mths_ex_med'].apply(np.log1p)

#X_test['acc_now_delinq'] = X_test['acc_now_delinq'].apply(np.log1p)

#X_test['tot_coll_amt'] = X_test['tot_coll_amt'].apply(np.log1p)

#X_test['collections_12_mths_ex_med']=np.floor(np.log2(X_test['collections_12_mths_ex_med']+1))
# 列の組み合わせを作ってみる。



X_train['grade_hmown'] = X_train['grade'].str.cat(X_train['home_ownership'])

X_test['grade_hmown'] = X_test['grade'].str.cat(X_test['home_ownership'])

X_train['grade_purpose'] = X_train['grade'].str.cat(X_train['purpose'])

X_test['grade_purpose'] = X_test['grade'].str.cat(X_test['purpose'])



X_train['subg_hmown'] = X_train['sub_grade'].str.cat(X_train['home_ownership'])

X_test['subg_hmown'] = X_test['sub_grade'].str.cat(X_test['home_ownership'])

X_train['subg_apptype'] = X_train['sub_grade'].str.cat(X_train['application_type'])

X_test['subg_apptype'] = X_test['sub_grade'].str.cat(X_test['application_type'])

X_train['subg_initstat'] = X_train['sub_grade'].str.cat(X_train['initial_list_status'])

X_test['subg_initstat'] = X_test['sub_grade'].str.cat(X_test['initial_list_status'])

X_train['subg_zip'] = X_train['sub_grade'].str.cat(X_train['zip_code'])

X_test['subg_zip'] = X_test['sub_grade'].str.cat(X_test['zip_code'])

X_train['subg_addstat'] = X_train['sub_grade'].str.cat(X_train['addr_state'])

X_test['subg_addstat'] = X_test['sub_grade'].str.cat(X_test['addr_state'])

X_train['subg_empln'] = X_train['sub_grade'].str.cat(X_train['emp_length'])

X_test['subg_empln'] = X_test['sub_grade'].str.cat(X_test['emp_length'])





X_train['hmown_initstat'] = X_train['home_ownership'].str.cat(X_train['initial_list_status'])

X_test['hmown_initstat'] = X_test['home_ownership'].str.cat(X_test['initial_list_status'])

X_train['hmown_addr_state'] = X_train['home_ownership'].str.cat(X_train['addr_state'])

X_test['hmown_addr_state'] = X_test['home_ownership'].str.cat(X_test['addr_state'])

X_train['hmown_apptype'] = X_train['home_ownership'].str.cat(X_train['application_type'])

X_test['hmown_apptype'] = X_test['home_ownership'].str.cat(X_test['application_type'])

X_train['hmown_zip'] = X_train['home_ownership'].str.cat(X_train['zip_code'])

X_test['hmown_zip'] = X_test['home_ownership'].str.cat(X_test['zip_code'])

#X_train['grade_subg_hmown'] = X_train['grade'].str.cat(X_train['sub_grade']).str.cat(X_train['home_ownership'])

#X_test['grade_subg_hmown'] = X_test['grade'].str.cat(X_test['sub_grade']).str.cat(X_train['home_ownership'])



X_train['subg_hmown_stat'] = X_train['subg_hmown'].str.cat(X_train['addr_state'])

X_test['subg_hmown_stat'] = X_test['subg_hmown'].str.cat(X_test['addr_state'])

X_test['subg_hmown_apptype'] = X_test['subg_hmown'].str.cat(X_test['application_type'])

X_train['subg_hmown_apptype'] = X_train['subg_hmown'].str.cat(X_train['application_type'])

X_test['subg_hmown_initstat'] = X_test['subg_hmown'].str.cat(X_test['initial_list_status'])

X_train['subg_hmown_initstat'] = X_train['subg_hmown'].str.cat(X_train['initial_list_status'])



X_train['subg_apptype_initial_list_status'] = X_train['subg_apptype'].str.cat(X_train['initial_list_status'])

X_test['subg_apptype_initial_list_status'] = X_test['subg_apptype'].str.cat(X_test['initial_list_status'])





X_train['subg_zip_addr_state'] = X_train['subg_zip'].str.cat(X_train['addr_state'])

X_test['subg_zip_addr_state'] = X_test['subg_zip'].str.cat(X_test['addr_state'])

X_train['subg_zip_application_type'] = X_train['subg_zip'].str.cat(X_train['application_type'])

X_test['subg_zip_application_type'] = X_test['subg_zip'].str.cat(X_test['application_type'])





X_train['hmown_initstat_addr_state'] = X_train['hmown_initstat'].str.cat(X_train['addr_state'])

X_test['hmown_initstat_addr_state'] = X_test['hmown_initstat'].str.cat(X_test['addr_state'])

X_train['hmown_initstat_application_type'] = X_train['hmown_initstat'].str.cat(X_train['application_type'])

X_test['hmown_initstat_application_type'] = X_test['hmown_initstat'].str.cat(X_test['application_type'])

X_train['hmown_initstat_zip_code'] = X_train['hmown_initstat'].str.cat(X_train['zip_code'])

X_test['hmown_initstat_zip_code'] = X_test['hmown_initstat'].str.cat(X_test['zip_code'])







X_test['subg_hmown_apptype_addr_state'] = X_test['subg_hmown_apptype'].str.cat(X_test['addr_state'])

X_train['subg_hmown_apptype_addr_state'] = X_train['subg_hmown_apptype'].str.cat(X_train['addr_state'])

X_test['subg_hmown_apptype_initial'] = X_test['subg_hmown_apptype'].str.cat(X_test['initial_list_status'])

X_train['subg_hmown_apptype_initial'] = X_train['subg_hmown_apptype'].str.cat(X_train['initial_list_status'])

X_test['subg_hmown_apptype_zip_code'] = X_test['subg_hmown_apptype'].str.cat(X_test['zip_code'])

X_train['subg_hmown_apptype_zip_code'] = X_train['subg_hmown_apptype'].str.cat(X_train['zip_code'])





X_train['subg_initstat_addstate'] = X_train['subg_initstat'].str.cat(X_train['addr_state'])

X_test['subg_initstat_addstate'] = X_test['subg_initstat'].str.cat(X_test['addr_state'])

X_train['subg_initstat_apptype'] = X_train['subg_initstat'].str.cat(X_train['application_type'])

X_test['subg_initstat_apptype'] = X_test['subg_initstat'].str.cat(X_test['application_type'])

X_train['subg_initstat_addr_state'] = X_train['subg_initstat'].str.cat(X_train['addr_state'])

X_test['subg_initstat_addr_state'] = X_test['subg_initstat'].str.cat(X_test['addr_state'])





#X_train['grade_hmown_count'] = X_train['grade'].str.cat(X_train['home_ownership']).map(X_train['grade'].str.cat(X_train['home_ownership']).value_counts())

#X_train['grade_subg_count'] = X_train['grade'].str.cat(X_train['sub_grade']).map(X_train['grade'].str.cat(X_train['sub_grade']).value_counts())

#X_train['grade_pups_count'] = X_train['grade'].str.cat(X_train['purpose']).map(X_train['grade'].str.cat(X_train['purpose']).value_counts())

#X_train['grade_zip_count'] = X_train['grade'].str.cat(X_train['zip_code']).map(X_train['grade'].str.cat(X_train['zip_code']).value_counts())



#X_train.loc[X_train['delinq_2yrs'] > 0, "delinq_2yrs_flg"]=1

#X_test.loc[df_test['delinq_2yrs'] > 0, "delinq_2yrs_flg"]=1

#X_train.loc[X_train['delinq_2yrs'] == 0, "delinq_2yrs_flg"]=0

#X_test.loc[df_test['delinq_2yrs'] == 0, "delinq_2yrs_flg"]=0





#このカウントエンコーディングは効果なし。

#X_train['zip_count'] = X_train['zip_code'].value_counts()

#X_test['zip_count'] = X_test['zip_code'].value_counts()

#X_train['addr_state_count'] = X_train['addr_state'].value_counts()

#X_test['addr_state_count'] = X_test['addr_state'].value_counts()

#summary = df_train['emp_title'].value_counts()

#X_train

#X_train['subg_dti'] = X_train['sub_grade'].map(X_train.groupby('sub_grade')['dti'].mean())

X_train

#summary = X_temp.groupby([col])[target].mean()

#X_test[col] = X_test[col].map(summary) 
#　文字列のカラムと数字のカラムのリストを抽出

cats = []

numcats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())

        

    else:

        numcats.append(col)
#　行ごとに欠損値の数を集計した列を追加

X_train['def_cn'] = X_train.isnull().sum(axis=1)

X_test['def_cn'] = X_test.isnull().sum(axis=1)
X_train[cats].fillna('#', inplace=True)

X_test[cats].fillna('#', inplace=True)



#　対数変換したカラムはリストから除外

numcats.remove('tot_cur_bal')

numcats.remove('inq_last_6mths')
#'''

# 正の値を取る変数を変換対象としてリストに格納

pos_cols = [col for col in numcats if min(X_train[col]) > 0 and

min(X_test[col]) > 0]



from sklearn.preprocessing import PowerTransformer

# 学習データに基づいて複数カラムのBox-Cox変換を定義

pt = PowerTransformer(method='box-cox')

pt.fit(X_train[pos_cols])

# 変換後のデータで各列を置換

X_train[pos_cols] = pt.transform(X_train[pos_cols])

X_test[pos_cols] = pt.transform(X_test[pos_cols])



#Yao-Johnson変換:

# 学習データに基づいて複数カラムのYeo-Johnson変換を定義

#pt = PowerTransformer(method='yeo-johnson')

#pt.fit(X_train[pos_cols])

# 変換後のデータで各列を置換

#X_train[pos_cols] = pt.transform(X_train[pos_cols])

#X_test[pos_cols] = pt.transform(X_test[pos_cols])

#'''
#　ここからテキスト変換にトライ

#　emp_title列をシリーズとして分離

TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()

TXT_train.fillna('#', inplace=True)

TXT_test.fillna('#', inplace=True)

#cv = CountVectorizer(min_df=2)

tfidf = TfidfVectorizer(max_features=1000, use_idf=True)


#TXT_train_enc = tfidf.fit_transform(TXT_train).todense()

#TXT_test_enc = tfidf.transform(TXT_test).todense()

TXT_train_enc = tfidf.fit_transform(TXT_train).todense()

TXT_test_enc = tfidf.transform(TXT_test).todense()

TXT_train_enc
#　配列を合計し、DataFrameに変換

TXT_train_enc_sum = pd.DataFrame(np.sum(TXT_train_enc,axis=1))

TXT_test_enc_sum = pd.DataFrame(np.sum(TXT_test_enc,axis=1))

#TXT_train_enc_sum = pd.DataFrame(TXT_train_enc)

#TXT_test_enc_sum = pd.DataFrame(TXT_test_enc)



#　列名を付与

TXT_train_enc_sum.columns=['emp_title_feat']

TXT_test_enc_sum.columns=['emp_title_feat']



####一旦この状態で

TXT_train_enc_sum
# 文字列をオーディナルエンコーディング

oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
# オーディナルエンコーディングした列をさらにターゲットエンコーディング・・・こっちのほうがスコアが良かったような。

#'''

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test[col] = X_test[col].map(summary) 

#    X_test[col + "_tgtenc"] = X_test[col].map(summary) 



    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#    skf = KFold(n_splits=5, random_state=71, shuffle=True)

#    skf = ShuffleSplit(n_splits=5, random_state=71)

    

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train[col]  = enc_train

#    X_train[col + "_tgtenc"]  = enc_train

#'''
# テキストエンコーディングしたものと元のDFに新たに連番を付与して、連番で結合。・・・テキストエンコーディングの過程でインデックスが新たに0からの連番に代わってしまったのでやむを得ず。



serial_num = pd.RangeIndex(start=0, stop=len(TXT_train_enc_sum.index), step=1)

TXT_train_enc_sum['No'] = serial_num

serial_num = pd.RangeIndex(start=0, stop=len(TXT_test_enc_sum.index), step=1)

TXT_test_enc_sum['No'] = serial_num

serial_num = pd.RangeIndex(start=0, stop=len(X_train.index), step=1)

X_train['No'] = serial_num

serial_num = pd.RangeIndex(start=0, stop=len(X_test.index), step=1)

X_test['No'] = serial_num



X_train=pd.merge(X_train, TXT_train_enc_sum)

X_test=pd.merge(X_test, TXT_test_enc_sum)
# 不要な列を削除

X_train = X_train.drop('No', axis=1)

X_test = X_test.drop('No', axis=1)

gc.collect()
display(X_train.shape)

display(X_test.shape)
# NaNは平均値で埋めとく

X_train.fillna(X_train.mean(), axis=0, inplace=True)

X_test.fillna(X_train.mean(), axis=0, inplace=True)
X_test.isnull().sum()
# 不要な列を削除（重要度が0.0）



delcols = ['acc_now_delinq','application_type','collections_12_mths_ex_med','pub_rec','delinq_2yrs','def_cn']





X_train = X_train.drop(delcols, axis=1)

X_test = X_test.drop(delcols, axis=1)

gc.collect()
#'''

# 全データで再学習し、testに対して予測する



y_pred_test = np.zeros(len(X_test))

SEEDNUM = 3

HOLDNUM = 5



skf = StratifiedKFold(n_splits=HOLDNUM, random_state=71, shuffle=True)



params = {"objective": 'binary',

          "boosting_type": "gbdt",

          "learning_rate": 0.11,

          "colsample_bytree":0.9,

          "num_leaves": 11,

          "max_depth":7,

#          "feature_fraction": 0.8,

#          "drop_rate": 0.1,

          "is_unbalance": False,

          "n_estimators":1000,

          'reg_alpha':0.0,

          'reg_lambda':0.0,

          'subsample_for_bin':20000,

          'subsample_freq':0,

          "min_child_samples": 200,

          "min_child_weight": 0.001,

          "min_split_gain": 2.0,

          "subsample": 1.0,

          "metric": "auc",

          "boost_from_average": False,

          }



gc.collect()

for s in range(10, 11 + SEEDNUM):

    params['seed'] = s

    

    y_pred_hold = np.zeros(len(X_test))



    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

        clf = LGBMClassifier(**params)



        clf.fit(X_train_, y_train_)

        y_pred_hold += clf.predict_proba(X_test)[:,1]



    y_pred_hold /= HOLDNUM

    y_pred_test += y_pred_hold



y_pred = y_pred_test / SEEDNUM



# さらにGBC

#GBclf = GradientBoostingClassifier()

#GBclf.fit(X_train, y_train)

#y_pred += GBclf.predict_proba(X_test)[:,1]

#y_pred = y_pred / 2

#'''
y_pred
submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')