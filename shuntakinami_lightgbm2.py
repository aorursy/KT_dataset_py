# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import scipy as sp

import pandas as pd

import sys

from pandas import DataFrame, Series

from pandas import plotting

from datetime import datetime

from sklearn import preprocessing, decomposition



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier # AdaBoost

from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import StackingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from hyperopt import fmin, tpe, hp, rand, Trials

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from lightgbm import LGBMClassifier
dtypes = {

    'ID': 'int32',

    'loan_amnt': 'int32',

    'installment': 'float32',

    'grade': 'object',

    'sub_grade': 'object',

    'emp_title': 'object',

    'emp_length': 'object',

    'home_ownership': 'object',

    'annual_inc': 'float32',

    'issue_d': 'object',

    'purpose': 'object',

    'title': 'object',

    'zip_code': 'object',

    'addr_state': 'object',

    'dti': 'float16',

    'delinq_2yrs': 'float32',

    'earliest_cr_line': 'object',

    'inq_last_6mths': 'float32',

    'mths_since_last_delinq': 'float32',

    'mths_since_last_record': 'float32',

    'open_acc': 'float32',

    'pub_rec': 'float32',

    'revol_bal': 'int32',

    'revol_util': 'float32',

    'total_acc': 'float32',

    'initial_list_status': 'object',

    'collections_12_mths_ex_med': 'float32',

    'mths_since_last_major_derog': 'float32',

    'application_type': 'object',

    'acc_now_delinq': 'float32',

    'tot_coll_amt': 'float32',

    'tot_cur_bal': 'float32',

    'loan_condition': 'int8'

}
df_train = pd.read_csv('/kaggle/input/homework-for-students3/train.csv',dtype=dtypes ,index_col=0)#, skiprows=lambda x: x%20!=0

df_test = pd.read_csv('/kaggle/input/homework-for-students3/test.csv',dtype=dtypes,index_col=0)# , skiprows=lambda x: x%20!=0

df_state= pd.read_csv('/kaggle/input/homework-for-students3/statelatlong.csv',index_col=0)

# df_spi=pd.read_csv('spi.csv',index_col=0)

# df_fzcode=pd.read_csv('free-zipcode-database.csv',index_col=0)
df_merged_train = pd.merge(df_train,df_state,left_on='addr_state',right_on='State',how='left',right_index=True)

df_merged_test = pd.merge(df_test,df_state,left_on='addr_state',right_on='State',how='left',right_index=True)

#使用データを区切る

df_merged_train=df_merged_train[pd.to_datetime(df_merged_train['issue_d']) >= datetime(2015,1,1)]
cats = []

for col in df_merged_train.columns:

    if df_merged_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, df_merged_train[col].nunique())

        

#count_enc

def count_enc(col,df):

    summary = df[col].value_counts()

    return df[col].map(summary)



def outlier_removing(df,n_sigma):



    for i in range(len(df.columns)):



        # 列を抽出する

        col = df.iloc[:,i]



        # 平均と標準偏差

        average = np.mean(col)

        sd = np.std(col)



        # 外れ値の基準点

        outlier_min = average - (sd) * n_sigma

        outlier_max = average + (sd) * n_sigma



        # 範囲から外れている値を除く

        col[col < outlier_min] = None

        col[col > outlier_max] = None



    return df
df_merged_train.isnull().sum()
#欠損値フラグ

df_merged_train.loc[df_merged_train['title'].isnull() == True , 'title_null_flg'] = 1

df_merged_train.loc[df_merged_train['title'].isnull() == False , 'title_null_flg'] = 0



df_merged_test.loc[df_merged_test['title'].isnull() == True , 'title_null_flg'] = 1

df_merged_test.loc[df_merged_test['title'].isnull() == False , 'title_null_flg'] = 0



df_merged_train.loc[df_merged_train['emp_title'].isnull() == True , 'emp_title_null_flg'] = 1

df_merged_train.loc[df_merged_train['emp_title'].isnull() == False , 'emp_title_null_flg'] = 0



df_merged_test.loc[df_merged_test['emp_title'].isnull() == True , 'emp_title_null_flg'] = 1

df_merged_test.loc[df_merged_test['emp_title'].isnull() == False , 'emp_title_null_flg'] = 0



df_merged_train.loc[df_merged_train['emp_length'].isnull() == True , 'emp_length_null_flg'] = 1

df_merged_train.loc[df_merged_train['emp_length'].isnull() == False , 'emp_length_null_flg'] = 0



df_merged_test.loc[df_merged_test['emp_length'].isnull() == True , 'emp_length_null_flg'] = 1

df_merged_test.loc[df_merged_test['emp_length'].isnull() == False , 'emp_length_null_flg'] = 0



df_merged_train.loc[df_merged_train['mths_since_last_delinq'].isnull() == True , 'mths_since_last_delinq_null_flg'] = 1

df_merged_train.loc[df_merged_train['mths_since_last_delinq'].isnull() == False , 'mths_since_last_delinq_null_flg'] = 0



df_merged_test.loc[df_merged_test['mths_since_last_delinq'].isnull() == True , 'mths_since_last_delinq_null_flg'] = 1

df_merged_test.loc[df_merged_test['mths_since_last_delinq'].isnull() == False , 'mths_since_last_delinq_null_flg'] = 0



df_merged_train.loc[df_merged_train['mths_since_last_record'].isnull() == True , 'mths_since_last_record_null_flg'] = 1

df_merged_train.loc[df_merged_train['mths_since_last_record'].isnull() == False , 'mths_since_last_record_null_flg'] = 0



df_merged_test.loc[df_merged_test['mths_since_last_record'].isnull() == True , 'mths_since_last_record_null_flg'] = 1

df_merged_test.loc[df_merged_test['mths_since_last_record'].isnull() == False , 'mths_since_last_record_null_flg'] = 0



df_merged_train.loc[df_merged_train['mths_since_last_major_derog'].isnull() == True , 'mths_since_last_major_derog_null_flg'] = 1

df_merged_train.loc[df_merged_train['mths_since_last_major_derog'].isnull() == False , 'mths_since_last_major_derog_null_flg'] = 0



df_merged_test.loc[df_merged_test['mths_since_last_major_derog'].isnull() == True , 'mths_since_last_major_derog_null_flg'] = 1

df_merged_test.loc[df_merged_test['mths_since_last_major_derog'].isnull() == False , 'mths_since_last_major_derog_null_flg'] = 0



df_merged_train.loc[df_merged_train['revol_util'].isnull() == True , 'revol_util'] = 1

df_merged_train.loc[df_merged_train['revol_util'].isnull() == False , 'revol_util'] = 0



df_merged_test.loc[df_merged_test['revol_util'].isnull() == True , 'revol_util'] = 1

df_merged_test.loc[df_merged_test['revol_util'].isnull() == False , 'revol_util'] = 0



# df_merged_train.loc[df_merged_train['dti'].isnull() == True , 'dti'] = 1

# df_merged_train.loc[df_merged_train['dti'].isnull() == False , 'dti'] = 0



# df_merged_test.loc[df_merged_test['dti'].isnull() == True , 'dti'] = 1

# df_merged_test.loc[df_merged_test['dti'].isnull() == False , 'dti'] = 0
#label

df_merged_train['application_type'] = df_merged_train['application_type'].map( {'Individual': 2, 'Joint App': 1} ).astype(int)

df_merged_test['application_type'] = df_merged_test['application_type'].map( {'Individual': 2, 'Joint App': 1} ).astype(int)



df_merged_train['initial_list_status'] = df_merged_train['initial_list_status'].map( {'f': 2, 'w': 1} ).astype(int)

df_merged_test['initial_list_status'] = df_merged_test['initial_list_status'].map( {'f': 2, 'w': 1} ).astype(int)



df_merged_train['grade']=df_merged_train['grade'].replace('A',1).replace('B',2).replace('C',3).replace('D',4).replace('E',5).replace('F',6).replace('G',7)

df_merged_test['grade']=df_merged_test['grade'].replace('A',1).replace('B',2).replace('C',3).replace('D',4).replace('E',5).replace('F',6).replace('G',7)



df_merged_train['sub_grade']=df_merged_train['sub_grade'].replace('A1',1).replace('A2',2).replace('A3',3).replace('A4',4).replace('A5',5).replace('B1',6).replace('B2',7).replace('B3',8).replace('B4',9).replace('B5',10).replace('C1',11).replace('C2',12).replace('C3',13).replace('C4',14).replace('C5',15).replace('D1',16).replace('D2',17).replace('D3',18).replace('D4',19).replace('D5',20).replace('E1',21).replace('E2',22).replace('E3',23).replace('E4',24).replace('E5',25).replace('F1',26).replace('F2',27).replace('F3',28).replace('F4',29).replace('F5',30).replace('G1',31).replace('G2',32).replace('G3',33).replace('G4',34).replace('G5',35)

df_merged_test['sub_grade']=df_merged_test['sub_grade'].replace('A1',1).replace('A2',2).replace('A3',3).replace('A4',4).replace('A5',5).replace('B1',6).replace('B2',7).replace('B3',8).replace('B4',9).replace('B5',10).replace('C1',11).replace('C2',12).replace('C3',13).replace('C4',14).replace('C5',15).replace('D1',16).replace('D2',17).replace('D3',18).replace('D4',19).replace('D5',20).replace('E1',21).replace('E2',22).replace('E3',23).replace('E4',24).replace('E5',25).replace('F1',26).replace('F2',27).replace('F3',28).replace('F4',29).replace('F5',30).replace('G1',31).replace('G2',32).replace('G3',33).replace('G4',34).replace('G5',35)



df_merged_train['emp_length']=df_merged_train['emp_length'].replace('10+ years',10).replace('9 years',9).replace('8 years',8).replace('7 years',7).replace('6 years',6).replace('5 years',5).replace('4 years',4).replace('3 years',3).replace('2 years',2).replace('1 year',1).replace('< 1 year',0).replace('n/a',-1)

df_merged_test['emp_length']=df_merged_test['emp_length'].replace('10+ years',10).replace('9 years',9).replace('8 years',8).replace('7 years',7).replace('6 years',6).replace('5 years',5).replace('4 years',4).replace('3 years',3).replace('2 years',2).replace('1 year',1).replace('< 1 year',0).replace('n/a',-1)
#Ordinal enc ##'purpose','addr_state'

col=['zip_code','home_ownership','addr_state','purpose','title']

oe = OrdinalEncoder(cols=col, return_df=True)

df_merged_train[col] = oe.fit_transform(df_merged_train[col])

df_merged_test[col] = oe.transform(df_merged_test[col])



# oe = OrdinalEncoder(cols='title', return_df=True)

# df_merged_train['title'] = oe.fit_transform(df_merged_train['title'])

# df_merged_test['title'] = oe.transform(df_merged_test['title'])
df_merged_train['weekday'] = pd.to_datetime(df_merged_train['issue_d']).dt.weekday

df_merged_test['weekday'] = pd.to_datetime(df_merged_test['issue_d']).dt.weekday



df_merged_train['month'] = pd.to_datetime(df_merged_train['issue_d']).dt.month

df_merged_test['month'] = pd.to_datetime(df_merged_test['issue_d']).dt.month



df_merged_train['day'] = pd.to_datetime(df_merged_train['issue_d']).dt.day

df_merged_test['day'] = pd.to_datetime(df_merged_test['issue_d']).dt.day
#欠損地処理

df_merged_train['mths_since_last_delinq'] = df_merged_train['mths_since_last_delinq'].fillna(0)

df_merged_test['mths_since_last_delinq'] = df_merged_test['mths_since_last_delinq'].fillna(0)

df_merged_train['mths_since_last_major_derog'] = df_merged_train['mths_since_last_major_derog'].fillna(0)

df_merged_test['mths_since_last_major_derog'] = df_merged_test['mths_since_last_major_derog'].fillna(0)

df_merged_train['mths_since_last_record'] = df_merged_train['mths_since_last_record'].fillna(0)

df_merged_test['mths_since_last_record'] = df_merged_test['mths_since_last_record'].fillna(0)



##職業フラグ

df_merged_train['emp_title_Consul_flg'] = df_merged_train['emp_title'].str.contains('Consultant', case=False).astype(float)

df_merged_test['emp_title_Consul_flg'] = df_merged_test['emp_title'].str.contains('Consultant', case=False).astype(float)



df_merged_train['emp_title_Accountant_flg'] = df_merged_train['emp_title'].str.contains('Accountant', case=False).astype(float)

df_merged_test['emp_title_Accountant_flg'] = df_merged_test['emp_title'].str.contains('Accountant', case=False).astype(float)



df_merged_train['emp_title_Account_flg'] = df_merged_train['emp_title'].str.contains('Account', case=False).astype(float)

df_merged_test['emp_title_Account_flg'] = df_merged_test['emp_title'].str.contains('Account', case=False).astype(float)



df_merged_train['emp_title_Finance_flg'] = df_merged_train['emp_title'].str.contains('Finance', case=False).astype(float)

df_merged_test['emp_title_Finance_flg'] = df_merged_test['emp_title'].str.contains('Finance', case=False).astype(float)



df_merged_train['emp_title_School_flg'] = df_merged_train['emp_title'].str.contains('School', case=False).astype(float)

df_merged_test['emp_title_School_flg'] = df_merged_test['emp_title'].str.contains('School', case=False).astype(float)



df_merged_train['emp_title_RN_flg'] = df_merged_train['emp_title'].str.match('RN', case=False).astype(float)

df_merged_test['emp_title_RN_flg'] = df_merged_test['emp_title'].str.match('RN', case=False).astype(float)



df_merged_train['emp_title_VP_flg'] = df_merged_train['emp_title'].str.match('VP', case=False).astype(float)

df_merged_test['emp_title_VP_flg'] = df_merged_test['emp_title'].str.match('VP', case=False).astype(float)



df_merged_train['emp_title_Business_flg'] = df_merged_train['emp_title'].str.contains('Business', case=False).astype(float)

df_merged_test['emp_title_Business_flg'] = df_merged_test['emp_title'].str.contains('Business', case=False).astype(float)



df_merged_train['emp_title_Warehouse_flg'] = df_merged_train['emp_title'].str.contains('Warehouse', case=False).astype(float)

df_merged_test['emp_title_Warehouse_flg'] = df_merged_test['emp_title'].str.contains('Warehouse', case=False).astype(float)



df_merged_train['emp_title_nurse_flg'] = df_merged_train['emp_title'].str.contains('nurse', case=False).astype(float)

df_merged_test['emp_title_nurse_flg'] = df_merged_test['emp_title'].str.contains('nurse', case=False).astype(float)



df_merged_train['emp_title_Supervisor_flg'] = df_merged_train['emp_title'].str.contains('Supervisor', case=False).astype(float)

df_merged_test['emp_title_Supervisor_flg'] = df_merged_test['emp_title'].str.contains('Supervisor', case=False).astype(float)



df_merged_train['emp_title_American_flg'] = df_merged_train['emp_title'].str.contains('American', case=False).astype(float)

df_merged_test['emp_title_American_flg'] = df_merged_test['emp_title'].str.contains('American', case=False).astype(float)



df_merged_train['emp_title_Academic_flg'] = df_merged_train['emp_title'].str.contains('Academic', case=False).astype(float)

df_merged_test['emp_title_Academic_flg'] = df_merged_test['emp_title'].str.contains('Academic', case=False).astype(float)



df_merged_train['emp_title_Manager_flg'] = df_merged_train['emp_title'].str.contains('Manager', case=False).astype(float)

df_merged_test['emp_title_Manager_flg'] = df_merged_test['emp_title'].str.contains('Manager', case=False).astype(float)



df_merged_train['emp_title_Officer_flg'] = df_merged_train['emp_title'].str.contains('Officer', case=False).astype(float)

df_merged_test['emp_title_Officer_flg'] = df_merged_test['emp_title'].str.contains('Officer', case=False).astype(float)



df_merged_train['emp_title_Inc_flg'] = df_merged_train['emp_title'].str.endswith('Inc.').astype(float)

df_merged_test['emp_title_Inc_flg'] = df_merged_test['emp_title'].str.endswith('Inc.').astype(float)



df_merged_train['emp_title_ASST_flg'] = df_merged_train['emp_title'].str.contains('ASST', case=False).astype(float)

df_merged_test['emp_title_ASST_flg'] = df_merged_test['emp_title'].str.contains('ASST', case=False).astype(float)



df_merged_train['emp_title_Director_flg'] = df_merged_train['emp_title'].str.contains('Director', case=False).astype(float)

df_merged_test['emp_title_Director_flg'] = df_merged_test['emp_title'].str.contains('Director', case=False).astype(float)



df_merged_train['emp_title_Assistant_flg'] = df_merged_train['emp_title'].str.contains('Assistant', case=False).astype(float)

df_merged_test['emp_title_Assistant_flg'] = df_merged_test['emp_title'].str.contains('Assistant', case=False).astype(float)



df_merged_train['emp_title_Administrative_flg'] = df_merged_train['emp_title'].str.contains('Administrative', case=False).astype(float)

df_merged_test['emp_title_Administrativet_flg'] = df_merged_test['emp_title'].str.contains('Administrative', case=False).astype(float)



df_merged_train['emp_title_Clerk_flg'] = df_merged_train['emp_title'].str.contains('Clerk', case=False).astype(float)

df_merged_test['emp_title_Clerk_flg'] = df_merged_test['emp_title'].str.contains('Clerk', case=False).astype(float)



df_merged_train['emp_title_president_flg'] = df_merged_train['emp_title'].str.contains('President', case=False).astype(float)

df_merged_test['emp_title_president_flg'] = df_merged_test['emp_title'].str.contains('President', case=False).astype(float)



df_merged_train.fillna(df_merged_train.median(),inplace=True)

df_merged_test.fillna(df_merged_train.median(),inplace=True)
#特徴量作成

# df_merged_train['subgrade_mul_grade']=df_merged_train['sub_grade']*df_merged_train['grade']

# df_merged_test['subgrade_mul_grade']=df_merged_test['sub_grade']*df_merged_test['grade']

# df_merged_train['subgrade_mul_length']=df_merged_train['sub_grade']*df_merged_train['emp_length']

# df_merged_test['subgrade_mul_length']=df_merged_test['sub_grade']*df_merged_test['emp_length']

df_merged_train['revbal_plus_loanamt']=df_merged_train['revol_bal']+df_merged_train['loan_amnt']

df_merged_test['revbal_plus_loanamt']=df_merged_test['revol_bal']+df_merged_test['loan_amnt']

df_merged_train['ownership_purpose']=df_merged_train['home_ownership']*df_merged_train['purpose']

df_merged_test['ownership_purpose']=df_merged_test['home_ownership']*df_merged_test['purpose']

df_merged_train['diff_issue']=(pd.to_datetime(df_merged_train['issue_d'])-pd.to_datetime(df_merged_train['earliest_cr_line'])).astype('timedelta64[D]')

df_merged_test['diff_issue']=(pd.to_datetime(df_merged_test['issue_d'])-pd.to_datetime(df_merged_test['earliest_cr_line'])).astype('timedelta64[D]')

df_merged_train['ratio_inc_inst']=df_merged_train['installment']/(df_merged_train['annual_inc']/12+df_merged_train['tot_cur_bal'])

df_merged_test['ratio_inc_inst']=df_merged_test['installment']/(df_merged_test['annual_inc']/12+df_merged_test['tot_cur_bal'])

df_merged_train['ratio_inc_rev']=df_merged_train['revol_bal']/(df_merged_train['annual_inc']/12+df_merged_train['tot_cur_bal'])

df_merged_test['ratio_inc_rev']=df_merged_test['revol_bal']/(df_merged_test['annual_inc']/12+df_merged_test['tot_cur_bal'])

df_merged_train['ratio_acc_deliq']=df_merged_train['acc_now_delinq']/df_merged_train['total_acc']

df_merged_test['ratio_acc_deliq']=df_merged_test['acc_now_delinq']/df_merged_test['total_acc']

df_merged_train['ratio_acc']=df_merged_train['open_acc']/df_merged_train['total_acc']

df_merged_test['ratio_acc']=df_merged_test['open_acc']/df_merged_test['total_acc']

df_merged_train['ratio_loan_inst']=df_merged_train['installment']/df_merged_train['loan_amnt']

df_merged_test['ratio_loan_inst']=df_merged_test['installment']/df_merged_test['loan_amnt']

# df_merged_train['ndelinq_mul_2yrsdeling']=df_merged_train['delinq_2yrs']*df_merged_train['acc_now_delinq']

# df_merged_test['ndelinq_mul_2yrsdeling']=df_merged_test['delinq_2yrs']*df_merged_test['acc_now_delinq']

#other 
#特徴量捨て

df_merged_train.drop(['emp_title'], axis=1, inplace=True)

df_merged_test.drop(['emp_title'], axis=1, inplace=True)

df_merged_train.drop(['City'], axis=1, inplace=True)

df_merged_test.drop(['City'], axis=1, inplace=True)

df_merged_train.drop(['earliest_cr_line'], axis=1, inplace=True)

df_merged_test.drop(['earliest_cr_line'], axis=1, inplace=True)

df_merged_train.drop(['issue_d'], axis=1, inplace=True)

df_merged_test.drop(['issue_d'], axis=1, inplace=True)

# df_merged_train.drop(['acc_now_delinq'], axis=1, inplace=True)

# df_merged_test.drop(['acc_now_delinq'], axis=1, inplace=True)
# df_merged_train[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']] = outlier_removing(df_merged_train[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']],3)

# df_merged_test[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']] = outlier_removing(df_merged_test[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']],3)

# X_train[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']] = np.log1p(X_train[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']].values)

# X_test[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']] = np.log1p(X_test[['loan_amnt','tot_coll_amt','tot_cur_bal','total_acc','installment','annual_inc']].loan_amnt.values)



# df_merged_train = df_merged_train.dropna(how='any',axis=0)

# df_merged_test = df_merged_test.dropna(how='any',axis=0)

#set train data

X_train = df_merged_train

y_train = X_train.loan_condition

X_train = X_train.drop(['loan_condition'], axis=1)

X_test = df_merged_test
%%time

scores = []

y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=10, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix]

#     clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

#         importance_type='split', learning_rate=0.05, max_depth=-1,

#         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#         n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#         random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

#         subsample=0.9, subsample_for_bin=200000, subsample_freq=0)

    clf = LGBMClassifier(boosting_type='gbdt', class_weight='balanced')

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(y_val, y_pred))

    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく



scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 10 # 最後にfold数で割る
# %%time

# # CVしてスコアを見てみる

# # なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

# scores = []

# y_pred=[]



# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#     X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#     X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

# #     clf=LGBMClassifier(boosting_type='gbdt', class_weight=None,

# #                colsample_bytree=0.7000000000000001, importance_type='split',

# #                learning_rate=0.05, max_depth=10, min_child_samples=20,

# #                min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,

# #                n_jobs=-1, num_leaves=31, objective=None, random_state=None,

# #                reg_alpha=0.0, reg_lambda=0.0, silent=True,

# #                subsample=0.9929417385040324, subsample_for_bin=200000,

# #                subsample_freq=0)

#     clf = LGBMClassifier(boostingtype = 'gbdt',classweight='balanced')

#     clf.fit(X_train_, y_train_)#,eval_metric='auc', eval_set=[(X_val, y_val)]

#     y_pred.append(clf.predict_proba(X_test)[:,1])

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

    

#     print('CV Score of Fold_%d is %f' % (i, score))

# y_pred=np.mean(y_pred)
#mean_score

print(np.mean(scores))

print(np.std(scores))

print(scores)



# 0.7102843670562476

# 0.002437473639420565

# [0.7095717  0.71136192 0.71034983 0.71279488 0.71252257 0.71254254

#  0.70422256 0.70916753 0.70887022 0.71143993]



# 0.7104757794238037

# 0.002475453493233583

# [0.70969072 0.71161023 0.7105387  0.71317503 0.71251474 0.71249956

#  0.70424479 0.70904828 0.70952367 0.71191207]
# 全データで再学習し、testに対して予測する



clf.fit(X_train, y_train)



y_pred=clf.predict_proba(X_test,num_iteration =clf.best_iteration_)[:,1]
submission = pd.read_csv('/kaggle/input/homework-for-students3/sample_submission.csv', index_col=0)#, skiprows=lambda x: x%20!=0



submission.loan_condition = y_pred

submission.to_csv('sample_submission.csv')
clf.booster_.feature_importance(importance_type='gain')
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')