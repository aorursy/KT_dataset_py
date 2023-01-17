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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import seaborn as sns



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import quantile_transform

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier



import gc



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
df_train = pd.read_csv('/kaggle/input/homework-for-students4plus/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/homework-for-students4plus/test.csv', index_col=0)



sample_submission = pd.read_csv('/kaggle/input/homework-for-students4plus/sample_submission.csv')

free_zipcde_database = pd.read_csv('/kaggle/input/homework-for-students4plus/free-zipcode-database.csv')

statelatlong = pd.read_csv('/kaggle/input/homework-for-students4plus/statelatlong.csv')

US_GDP_by_State = pd.read_csv('/kaggle/input/homework-for-students4plus/US_GDP_by_State.csv')
df_train.shape, df_test.shape
# statelatlong と結合

df_train = pd.merge(df_train, statelatlong, how='left', left_on='addr_state', right_on='State').set_index(df_train.index)

df_test = pd.merge(df_test, statelatlong, how='left', left_on='addr_state', right_on='State').set_index(df_test.index)
df_train.shape, df_test.shape
US_GDP_by_State_grpby = US_GDP_by_State.groupby("State")[['State & Local Spending','Gross State Product','Real State Growth %','Population (million)']].mean()

US_GDP_by_State_grpby.head()
US_GDP_by_State_grpby['State & Local Spending/Population'] = US_GDP_by_State_grpby['State & Local Spending']/US_GDP_by_State_grpby['Population (million)']

US_GDP_by_State_grpby['Gross State Product/Population'] = US_GDP_by_State_grpby['Gross State Product']/US_GDP_by_State_grpby['Population (million)']

US_GDP_by_State_grpby = US_GDP_by_State_grpby.drop(['State & Local Spending'], axis=1)

US_GDP_by_State_grpby = US_GDP_by_State_grpby.drop(['Gross State Product'], axis=1)

US_GDP_by_State_grpby = US_GDP_by_State_grpby.drop(['Population (million)'], axis=1)

US_GDP_by_State_grpby.head()
# US_GDP_by_State と結合

df_train = pd.merge(df_train, US_GDP_by_State_grpby, how='left', left_on='City', right_on='State').set_index(df_train.index)

df_test = pd.merge(df_test, US_GDP_by_State_grpby, how='left', left_on='City', right_on='State').set_index(df_test.index)
df_train.shape, df_test.shape
for col in df_train.columns:

        print(col, df_train[col].nunique(), df_train[col].dtype)
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

nums = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        print('cats:', col, df_train[col].nunique(), df_train[col].dtype)

    else:

        nums.append(col)

        print('nums:',col, df_train[col].nunique(), df_train[col].dtype)
df_train.isnull().sum()
#各列の構成要素の確認

for i in df_train.columns:

    print(df_train[i].value_counts())

    print('\n')
# df_train['installment'] = df_train['installment'].apply(np.log1p)

# df_train['annual_inc'] = df_train['annual_inc'].apply(np.log1p)

# df_train['dti'] = df_train['dti'].apply(np.log1p)



# df_test['installment'] = df_test['installment'].apply(np.log1p)

# df_test['annual_inc'] = df_test['annual_inc'].apply(np.log1p)

# df_test['dti'] = df_test['dti'].apply(np.log1p)
fig=plt.figure(figsize=[20,60])



target = 'loan_condition'



i = 0

for col in nums:

    i=i+1

    if col == target:

        continue



    else:

        plt.subplots_adjust(wspace=0.2, hspace=0.8)

        ax_name = fig.add_subplot(40,2,i)

        ax_name.hist(df_train[col],bins=30,density=True, alpha=0.5,color = 'r')

        ax_name.hist(df_test[col],bins=30,density=True, alpha=0.5, color = 'b')

        ax_name.set_title(col)
import datetime

df_train['issue_d'] = pd.to_datetime(df_train['issue_d'], format='%b-%Y')

df_test['issue_d'] = pd.to_datetime(df_test['issue_d'], format='%b-%Y')
# issue_d で集計して、時系列のデータの推移を確認してみる

df_train_m = df_train.groupby('issue_d').agg(np.sum).copy()

df_test_m = df_test.groupby('issue_d').agg(np.sum).copy()
plt.plot(df_train_m['loan_condition'])

plt.title('loan_condition')

plt.xlabel('issue_d')

plt.ylabel('loan_condition')

plt.show()
fig=plt.figure(figsize=[20,60])



target = 'loan_condition'



i = 0

for col in nums:

    i=i+1

    if col == target:

        continue



    else:

        plt.subplots_adjust(wspace=0.2, hspace=0.8)

        ax_name = fig.add_subplot(40,2,i)



        plt.plot(df_train_m[col])

        plt.plot(df_test_m[col])

        plt.title(col)

        plt.xlabel('issue_d')

        plt.ylabel(col)

        plt.show()
# 2014年以降のデータに限定

df_train['year'] = df_train['issue_d'].dt.year



#df_train.drop(df_train.index[df_train['year'] <= 2013], inplace=True) # 2014-2015年を訓練データとする

df_train.drop(df_train.index[df_train['year'] <= 2014], inplace=True) # 2015年を訓練データとする
df_train.shape, df_test.shape
# drop 後の再確認

df_train_m = df_train.groupby('issue_d').agg(np.sum).copy()

df_test_m = df_test.groupby('issue_d').agg(np.sum).copy()
fig=plt.figure(figsize=[20,60])



target = 'loan_condition'



i = 0

for col in nums:

    i=i+1

    if col == target:

        plt.subplots_adjust(wspace=0.2, hspace=0.8)

        ax_name = fig.add_subplot(40,2,i)



        plt.plot(df_train_m[col])

        plt.title(col)

        plt.xlabel('issue_d')

        plt.ylabel(col)

        plt.xticks(rotation=45)

        plt.show()



    else:

        plt.subplots_adjust(wspace=0.2, hspace=0.8)

        ax_name = fig.add_subplot(40,2,i)



        plt.plot(df_train_m[col])

        plt.plot(df_test_m[col])

        plt.title(col)

        plt.xlabel('issue_d')

        plt.ylabel(col)

        plt.xticks(rotation=45)

        plt.show()
#df_train[df_train.emp_title.isnull()==True]

df_train['emp_title'].isnull().sum()
# emp_titleのnullフラグを作成(このフラグは重要)

df_train['emp_title_null_flg'] = np.where(df_train['emp_title'].isnull()==True, 1, 0)

df_test['emp_title_null_flg'] = np.where(df_test['emp_title'].isnull()==True, 1, 0)
df_train['emp_title_null_flg'].value_counts()
# df_train.query('emp_title == ["president", "CEO"]')
# df_train['emp_title_president_flg'] = np.where((df_train['emp_title'] == 'president') | (df_train['emp_title'] == 'CEO'), 1, 0)

# df_test['emp_title_president_flg'] = np.where((df_test['emp_title'] == 'president') | (df_test['emp_title'] == 'CEO'), 1, 0)
# df_train['emp_title_president_flg'].value_counts()
# # debt_consolidation_flgの作成

# df_train['purpose'].value_counts()
# df_train['debt_consolidation_flg'] = np.where((df_train['purpose'] == 'debt_consolidation'), 1, 0)

# df_test['debt_consolidation_flg'] = np.where((df_test['purpose'] == 'debt_consolidation'), 1, 0)
# df_train['debt_consolidation_flg'].value_counts()
# Textをよけておく

TXT_train = df_train.emp_title.copy()

TXT_test = df_test.emp_title.copy()



#df_train = df_train.drop(['emp_title'], axis=1)

#df_test = df_test.drop(['emp_title'], axis=1)
df_train = df_train.drop(['year'], axis=1)

df_train = df_train.drop(['issue_d'], axis=1)

#df_train = df_train.drop(['title'], axis=1) 

df_train = df_train.drop(['earliest_cr_line'], axis=1)

#df_train = df_train.drop(['State'], axis=1) # adder_stateとかぶっている。かつ、adder_stateの方がfeature importanceが上。

#df_train = df_train.drop(['City'], axis=1) # adder_stateとかぶっている。かつ、adder_stateの方がfeature importanceが上。

#df_train = df_train.drop(['addr_state'], axis=1) # trainとtest で誤差が大きい。zip_codeで十分



df_test = df_test.drop(['issue_d'], axis=1)

#df_test = df_test.drop(['title'], axis=1)

df_test = df_test.drop(['earliest_cr_line'], axis=1)

#df_test = df_test.drop(['State'], axis=1)

#df_test = df_test.drop(['City'], axis=1)

#df_test = df_test.drop(['addr_state'], axis=1) # trainとtest で誤差が大きい。zip_codeで十分

df_train.shape, df_test.shape
X_train = df_train.drop(['loan_condition'], axis=1)

y_train = df_train.loan_condition



X_test = df_test
X_train.isnull().sum()
X_test.isnull().sum()
#emp_lengthが欠損している人のgrade。emp_lengthの欠損値補完は、gradeでgroup byした際の平均値がいいかも？

X_train[X_train['emp_length'].isnull() == True].grade.value_counts()

def mapping(map_col, mapping):

    X_train[map_col] = X_train[map_col].map(mapping)

    X_test[map_col] = X_test[map_col].map(mapping)
grade_mapping = { "A": 1,"B": 2,"C": 3,"D": 4,"E": 5,"F": 6,"G": 7 }



subgrade_mapping = {"A1": 1,"A2": 2,"A3": 3,"A4": 4,"A5": 5,"B1": 6,"B2": 7,"B3": 8,"B4": 9,"B5": 10,

                    "C1": 11,"C2": 12,"C3": 13,"C4": 14,"C5": 15,"D1": 16,"D2": 17,"D3": 18,"D4": 19,"D5": 20,

                    "E1": 21,"E2": 22,"E3": 23,"E4": 24,"E5": 25,"F1": 26,"F2": 27,"F3": 28,"F4": 29,"F5": 30,

                    "G1": 31,"G2": 32,"G3": 33,"G4": 34,"G5": 35

                   }



emp_length_mapping = {"< 1 year": 0.5, "1 year": 1, "2 year": 2,"3 year": 3,"4 year": 4,"5 year": 5,

                      "6 year": 6, "7 year": 7, "8 year": 8,"9 year": 9,"10+ years": 10

                    }
mapping('grade', grade_mapping)

mapping('sub_grade', subgrade_mapping)

mapping('emp_length', emp_length_mapping)
#調査
#emp_lengthが0の人はどんな人？

X_train[X_train['emp_length'].isnull() == True].head(100)
#emp_lengthが欠損している人のgrade。emp_lengthの欠損値補完は、gradeでgroup byした際の平均値がいいかも？

X_train[X_train['emp_length'].isnull() == True].grade.value_counts()
X_train[X_train['dti'].isnull()==True]
X_train[X_train['revol_util'].isnull()==True]
X_train[X_train['mths_since_last_delinq'].isnull()==True]
X_train[X_train['mths_since_last_major_derog'].isnull()==True]
# 欠損値の補完処理
# # nullフラグを作成　#このフラグはないほうがいい

# X_train['mths_since_last_delinq_null_flg'] = np.where(X_train['mths_since_last_delinq'].isnull()==True, 1, 0)

# X_test['mths_since_last_delinq_null_flg'] = np.where(X_test['mths_since_last_delinq'].isnull()==True, 1, 0)



# X_train['mths_since_last_record_null_flg'] = np.where(X_train['mths_since_last_record'].isnull()==True, 1, 0)

# X_test['mths_since_last_record_null_flg'] = np.where(X_test['mths_since_last_record'].isnull()==True, 1, 0)



# X_train['mths_since_last_major_derog_null_flg'] = np.where(X_train['mths_since_last_major_derog'].isnull()==True, 1, 0)

# X_test['mths_since_last_major_derog_null_flg'] = np.where(X_test['mths_since_last_major_derog'].isnull()==True, 1, 0)


X_train['mths_since_last_delinq'].fillna(0, inplace=True)

X_train['mths_since_last_record'].fillna(0, inplace=True)

X_train['mths_since_last_major_derog'].fillna(0, inplace=True)

X_train['dti'].fillna(0, inplace=True)



X_test['mths_since_last_delinq'].fillna(0, inplace=True)

X_test['mths_since_last_record'].fillna(0, inplace=True)

X_test['mths_since_last_major_derog'].fillna(0, inplace=True)

X_test['inq_last_6mths'].fillna(0, inplace=True)

X_test['dti'].fillna(0, inplace=True)
#sub_gradeのgroupbyした平均値で欠損値を補完する

X_train['revol_util'] = X_train.groupby(['sub_grade'])['revol_util'].apply(lambda d: d.fillna(d.mean()))

X_train['emp_length'] = X_train.groupby(['sub_grade'])['emp_length'].apply(lambda d: d.fillna(d.mean()))



X_test['revol_util'] = X_test.groupby(['sub_grade'])['revol_util'].apply(lambda d: d.fillna(d.mean()))

X_test['emp_length'] = X_test.groupby(['sub_grade'])['emp_length'].apply(lambda d: d.fillna(d.mean()))
# 四則演算による特徴量の作成(1)

X_train['grade+subg'] = X_train['grade'] + X_train['sub_grade']

X_test['grade+subg'] = X_test['grade'] + X_test['sub_grade']
# 四則演算による特徴量の追加(2)

X_train['loan_amnt/installment'] = X_train['loan_amnt'] / X_train['installment']

X_train['revol_bal/installment'] = X_train['revol_bal'] / X_train['installment']

#X_train['tot_cur_bal/loan_amnt'] = X_train['tot_cur_bal'] / X_train['loan_amnt']



X_test['loan_amnt/installment'] = X_test['loan_amnt'] / X_test['installment']

X_test['revol_bal/installment'] = X_test['revol_bal'] / X_test['installment']

#X_test['tot_cur_bal/loan_amnt'] = X_test['tot_cur_bal'] / X_test['loan_amnt']

# 四則演算による特徴量の追加(3)

cols=['annual_inc','dti', 'emp_length', 'loan_amnt', 'open_acc','emp_title_null_flg', 

      'inq_last_6mths', 'mths_since_last_record', 'revol_bal', 'revol_util', 'tot_cur_bal']



for col in cols:

    X_train['*sub_grade_' + col] = X_train[col] * X_train['sub_grade']

    X_train['*grade_' + col] = X_train[col] * X_train['grade'] 

    

    X_test['*sub_grade_' + col] = X_test[col] * X_test['sub_grade']

    X_test['*grade_' + col] = X_test[col] * X_test['grade']
cats = []

nums = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        print('cats:', col, X_train[col].nunique(), X_train[col].dtype)

    else:

        nums.append(col)

        print('nums:',col, X_train[col].nunique(), X_train[col].dtype)
# 正規分布に従っていないのでランクガウスを実施する



X_all = pd.concat([X_train, X_test], axis=0)

X_all[nums] = quantile_transform(X_all[nums], n_quantiles=100, random_state=0, output_distribution='normal')
X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]
# カウントエンコーディング

#ce_cols = ['grade','sub_grade','home_ownership','purpose','zip_code','initial_list_status','application_type']

ce_cols = ['purpose']



for col in ce_cols:

    train_summary = X_train[col].value_counts()

    test_summary = X_test[col].value_counts()



    # mapする。

    X_train['ce_' + col] = X_train[col].map(train_summary)

    X_test['ce_' + col] = X_test[col].map(test_summary)
# Ordinal Encoder は、登場順序の順番。label encoder は、辞書の順に整数値を割り当てる。

#oe_cols = ['home_ownership', 'purpose','zip_code','initial_list_status','application_type']

oe_cols = ['home_ownership', 'purpose','zip_code','initial_list_status','application_type', 'addr_state', 'State', 'City', 'emp_title', 'title']

encoder = OrdinalEncoder(cols=oe_cols)

X_train[oe_cols] = encoder.fit_transform(X_train[oe_cols])

X_test[oe_cols] = encoder.transform(X_test[oe_cols])
# Target Encoding

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



#te_cols = ['grade','sub_grade','home_ownership','purpose','zip_code','initial_list_status','application_type', 'emp_title_null_flg', 'emp_title_president_flg']

#te_cols = ['home_ownership','purpose','zip_code','initial_list_status','application_type', 'Latitude', 'Longitude']

#te_cols = ['grade','sub_grade','emp_length', 'home_ownership','purpose','zip_code','initial_list_status','application_type', 'Latitude', 'Longitude']

te_cols = ['home_ownership','purpose','zip_code','initial_list_status','Latitude', 'Longitude', 'addr_state', 'State', 'City', 'emp_title', 'emp_length', 'title']





#for col in cats: #te_cols:

for col in te_cols:    



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test['te_' + col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train['te_' + col]  = enc_train
# 重要度の高い特徴量でエンコードを実施(1)

target = 'sub_grade'

X_temp = pd.concat([X_train, y_train], axis=1)



#fe_cols = ['grade','sub_grade','home_ownership','purpose','zip_code','initial_list_status','application_type', 'emp_title_null_flg', 'emp_title_president_flg']

fe_cols = ['home_ownership','purpose','zip_code','initial_list_status','application_type']

#fe_cols = ['home_ownership','purpose','zip_code']



for col in fe_cols:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test['fe_' + col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train['fe_' + col]  = enc_train
# 重要度の高い特徴量でエンコードを実施(2)

target = 'grade'

X_temp = pd.concat([X_train, y_train], axis=1)



fe_cols = ['home_ownership','purpose','zip_code','initial_list_status','application_type']

#fe_cols = ['home_ownership','purpose','zip_code']



for col in fe_cols:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test['fe2_' + col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train['fe2_' + col]  = enc_train
# 重要度の高い特徴量でエンコードを実施(3)

target = 'grade+subg'

X_temp = pd.concat([X_train, y_train], axis=1)



fe_cols = ['home_ownership','purpose','zip_code','initial_list_status','application_type']

#fe_cols = ['home_ownership','purpose','zip_code']



for col in fe_cols:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test['fe3_' + col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train['fe3_' + col]  = enc_train
X_train
X_train.shape, X_test.shape
fig=plt.figure(figsize=[20,60])



i = 0

#for col in nums:

for col in X_train.columns:

    i=i+1

    

    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(90,2,i)

    ax_name.hist(X_train[col],bins=30,density=True, alpha=0.5,color = 'r')

    ax_name.hist(X_test[col],bins=30,density=True, alpha=0.5, color = 'b')

    ax_name.set_title(col)
X_train.isnull().sum()
X_test.isnull().sum()
# LightGBM GroupKFold

groups = X_train.zip_code.values



y_pred_dbdt_avg = np.zeros(len(X_test))

num_split = 5



gkf = GroupKFold(n_splits = num_split)

scores = []

scores2 = []



iter_num = 6 # seed averagingで繰り返す回数を設定。実際には、（iter_num-1）回



for random_state in range(1, iter_num): # iter_num の回数分、繰り返す

    for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):



        X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

        X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]



        lgbm_clf = LGBMClassifier(objective='binary',

                             n_estimators=10000, 

                             boosting_type='gbdt',

                             importance_type='gain',

                             class_weight='balanced',

                             learning_rate=0.05,

                             max_depth=10,

                             num_leaves=31,

                             subsample=0.99,

                             colsample_bytree=0.70,

                             min_child_samples=20,

                             lambda_l1=0.91,

                             lambda_l2=0.61,

                             random_state=random_state, 

                             silent = 1,

                             n_jobs=-1)



        lgbm_clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', verbose=300, eval_set=[(X_val, y_val)])

    

        y_pred = lgbm_clf.predict_proba(X_val)[:,1]

        score2 = roc_auc_score(y_train_, lgbm_clf.predict_proba(X_train_)[:,1])

        scores2.append(score2)

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)



        print('train CV Score of Fold_%d is %f' % (i, score2))

        print('---------------------------------------------------')

        print('val CV Score of Fold_%d is %f' % (i, score))

        print('---------------------------------------------------')

        print('diff', score - score2)

        print('===================================================')

    

        # 予測確率の平均値を求める

        y_pred_dbdt_avg += lgbm_clf.predict_proba(X_test)[:,1]



y_pred_dbdt_avg = y_pred_dbdt_avg / (num_split * (iter_num - 1))



print('Average')

print('val avg:', np.mean(scores))

print(scores)

print('=============================')

print('train avg:', np.mean(scores2))

print(scores2)

print('=============================')

print('diff',np.mean(scores) - np.mean(scores2))

# # LightGBM SKF



# y_pred_dbdt_avg2 = np.zeros(len(X_test))

# num_split = 5



# skf = StratifiedKFold(n_splits=num_split, random_state=71, shuffle=True)

# scores = []

# scores2 = []



# iter_num = 6 # seed averagingで繰り返す回数を設定。実際には、（iter_num-1）回



# for random_state in range(1, iter_num): # iter_num の回数分、繰り返す

#     for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):



#         X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#         X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



#         lgbm_clf = LGBMClassifier(objective='binary',

#                              n_estimators=10000, 

#                              boosting_type='gbdt',

#                              importance_type='gain',

#                              class_weight='balanced',

#                              learning_rate=0.05,

#                              max_depth=10,

#                              num_leaves=31,

#                              subsample=0.99,

#                              colsample_bytree=0.70,

#                              min_child_samples=20,

#                              lambda_l1=0.91,

#                              lambda_l2=0.61,

#                              random_state=random_state, 

#                              silent = 1,

#                              n_jobs=-1)



#         lgbm_clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', verbose=300, eval_set=[(X_val, y_val)])

    

#         y_pred = lgbm_clf.predict_proba(X_val)[:,1]

#         score2 = roc_auc_score(y_train_, lgbm_clf.predict_proba(X_train_)[:,1])

#         scores2.append(score2)

#         score = roc_auc_score(y_val, y_pred)

#         scores.append(score)



#         print('train CV Score of Fold_%d is %f' % (i, score2))

#         print('---------------------------------------------------')

#         print('val CV Score of Fold_%d is %f' % (i, score))

#         print('---------------------------------------------------')

#         print('diff', score - score2)

#         print('===================================================')

    

#         # 予測確率の平均値を求める

#         y_pred_dbdt_avg2 += lgbm_clf.predict_proba(X_test)[:,1]



# y_pred_dbdt_avg2 = y_pred_dbdt_avg2 / (num_split * (iter_num - 1))



# print('Average')

# print('val avg:', np.mean(scores))

# print(scores)

# print('=============================')

# print('train avg:', np.mean(scores2))

# print(scores2)

# print('=============================')

# print('diff',np.mean(scores) - np.mean(scores2))

# Plot feature importance

# Initialize an empty array to hold feature importances

feature_importances = np.zeros(X_train.shape[1])



importances = lgbm_clf.feature_importances_



indices = np.argsort(importances)[::-1]



feat_labels = X_train.columns[0:]

for f in range(X_train.shape[1]):

    IMPORTANCES_LIST = pd.DataFrame([["%2d) %-*s %f" % (f +1, 30, feat_labels[indices[f]], importances[indices[f]])]] )

#    IMPORTANCES_LIST.to_csv('./data/feature_importances.csv', mode='a', header=False, index=False)

    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
import lightgbm as lgb

fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(lgbm_clf, max_num_features=100, ax=ax, importance_type='gain')
# TargetEncording後の欠損値補完

X_train.fillna(0, axis=0, inplace=True)

X_test.fillna(0, axis=0, inplace=True)
# X_train = X_train.drop(['emp_title'], axis=1)

# X_test = X_test.drop(['emp_title'], axis=1)
fig=plt.figure(figsize=[20,60])



target = 'loan_condition'



i = 0

for col in X_train.columns:

    i=i+1



    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(90,2,i)

    ax_name.hist(X_train[col],bins=30,density=True, alpha=0.5,color = 'r')

    ax_name.hist(X_test[col],bins=30,density=True, alpha=0.5, color = 'b')

    ax_name.set_title(col)
col = X_train.columns



scaler = StandardScaler()

scaler.fit(X_train)



X_train[col] = scaler.transform(X_train[col])

X_test[col] = scaler.transform(X_test[col])
X_train
# # GradientBoost

# from sklearn.model_selection import train_test_split



# X_train_, X_val, y_train_, y_val = train_test_split(X_train,

#                                                     y_train,

#                                                     test_size=0.25,

#                                                     shuffle=False,

#                                                     random_state=1)



# gbc_clf = GradientBoostingClassifier(n_estimators=90, learning_rate=0.25, random_state=1)



# gbc_clf.fit(X_train_, y_train_)

# y_pred_gbc = gbc_clf.predict_proba(X_val)[:,1]

# score2 = roc_auc_score(y_train_, gbc_clf.predict_proba(X_train_)[:,1])

# score = roc_auc_score(y_val, y_pred_gbc)



# y_pred_gbc_avg = gbc_clf.predict_proba(X_test)[:,1]



# print('train Score:', score2)

# print('---------------------------------------------------')

# print('val Score:', score)

# print('---------------------------------------------------')

# print('diff', score - score2)

from tensorflow.keras.layers import Dense ,Dropout, BatchNormalization, Input, Embedding, SpatialDropout1D, Reshape, Concatenate, Activation

from tensorflow.keras.optimizers import Adam

from keras import optimizers

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import AUC

from sklearn.preprocessing import StandardScaler



# Import Keras packages

from keras.models import Sequential, load_model

from keras.wrappers.scikit_learn import KerasRegressor

from keras.wrappers.scikit_learn import KerasClassifier

from keras import regularizers

# from sklearn.model_selection import train_test_split

# X_train_, X_val, y_train_, y_val = train_test_split(X_train,

#                                                     y_train,

#                                                     test_size=0.3,

#                                                     shuffle=False,

#                                                     random_state=1)
# create regression model

from keras import regularizers

weight_decay = 0.01



model = Sequential()

    

model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))



model.add(Dense(32, kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))



model.add(Dense(16, kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))



model.add(Dense(8, kernel_initializer='he_normal'))

model.add(Dropout(0.4))



model.add(Dense(1, activation='sigmoid'))

    



# compile model

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[AUC()])

 
# Learning

from keras.callbacks import EarlyStopping, ModelCheckpoint



#callback_es = EarlyStopping(monitor='val_loss', patience=10)

callback_es = EarlyStopping(monitor='val_loss', patience=20)

#callback_op = ModelCheckpoint(filepath='weights.{epoch:02d}.hdf5')



history = model.fit(X_train, y_train, epochs=150, batch_size=128, validation_split=0.50, verbose=1, callbacks=[callback_es])

y_pred_keras = model.predict(X_test).reshape(-1,)
## [test]

# y_pred_keras

# (y_pred_keras + y_pred_keras)/2
# LogisticRegression SKF



y_pred_lr_avg = np.zeros(len(X_test))

num_split = 5



skf = StratifiedKFold(n_splits=num_split, random_state=71, shuffle=True)

scores = []

scores2 = []



iter_num = 6 # seed averagingで繰り返す回数を設定。実際には、（iter_num-1）回



for random_state in range(1, iter_num): # iter_num の回数分、繰り返す

    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):



        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        lr_clf = LogisticRegression(l1_ratio=0.9, class_weight='balanced', random_state=random_state)



        lr_clf.fit(X_train_, y_train_)

    

        y_pred = lr_clf.predict_proba(X_val)[:,1]

        score2 = roc_auc_score(y_train_, lr_clf.predict_proba(X_train_)[:,1])

        scores2.append(score2)

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)



        print('train CV Score of Fold_%d is %f' % (i, score2))

        print('---------------------------------------------------')

        print('val CV Score of Fold_%d is %f' % (i, score))

        print('---------------------------------------------------')

        print('diff', score - score2)

        print('===================================================')

    

        # 予測確率の平均値を求める

        y_pred_lr_avg += lr_clf.predict_proba(X_test)[:,1]



y_pred_lr_avg = y_pred_lr_avg / (num_split * (iter_num - 1))



print('Average')

print('val avg:', np.mean(scores))

print(scores)

print('=============================')

print('train avg:', np.mean(scores2))

print(scores2)

print('=============================')

print('diff',np.mean(scores) - np.mean(scores2))

# TEXT 特徴量の追加

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 2))



train = tfidf.fit_transform(TXT_train.fillna('#'))

test = tfidf.transform(TXT_test.fillna('#'))



#X_train = sp.sparse.hstack([X_train, train]).todense() # 結合

#X_test = sp.sparse.hstack([X_test, test]).todense()



X_train = sp.sparse.hstack([X_train, train])# 結合

X_test = sp.sparse.hstack([X_test, test])



X_train = X_train.tocsr()# 行方向のスライスができるように変換する

X_test = X_test.tocsr()



#del TXT_train, TXT_test

gc.collect()

X_train.shape, X_test.shape
num_train = int(X_train.shape[0]*0.7)



X_train_ = X_train[:num_train, :]

y_train_ = y_train[:num_train]



X_val = X_train[num_train:, :]

y_val = y_train[num_train:]
# Textを追加したモデル

lgbm_clf = LGBMClassifier(objective='binary',

                          n_estimators=10000, 

                          boosting_type='gbdt',

                          importance_type='gain',

                          class_weight='balanced',

                          learning_rate=0.05,

                          max_depth=10,

                          num_leaves=31,

                          subsample=0.99,

                          colsample_bytree=0.70,

                          min_child_samples=20,

                          lambda_l1=0.91,

                          lambda_l2=0.61,

                          random_state=71, 

                          silent = 1,

                          n_jobs=-1)



lgbm_clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', verbose=300, eval_set=[(X_val, y_val)])



y_pred_txt = lgbm_clf.predict_proba(X_val)[:,1]



print(roc_auc_score(y_val, y_pred_txt)) # 検定スコア



y_pred_dbdt_txt = lgbm_clf.predict_proba(X_test)[:,1]
# # 全データで再学習し、testに対して予測する: averageをとるので不要

# clf.fit(X_train, y_train)



# y_pred = clf.predict_proba(X_test)[:,1]





# # sample submissionを読み込んで、予測値を代入の後、保存する

# submission = pd.read_csv('/kaggle/input/homework-for-students4plus/sample_submission.csv', index_col=0)



# submission.loan_condition = pred_test

# submission.to_csv('./submission.csv')



# submission.head()
#pred_test = (1*y_pred_dbdt_avg + 0*y_pred_dbdt_txt+ 0*y_pred_lr_avg+ 0*y_pred_keras) #0.70088

pred_test = (0.50*y_pred_dbdt_avg + 0.10*y_pred_dbdt_txt+ 0.20*y_pred_lr_avg+ 0.20*y_pred_keras)

pred_test
submission = pd.read_csv('/kaggle/input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = pred_test

submission.to_csv('submission.csv')