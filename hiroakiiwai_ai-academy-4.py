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

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

df_train = pd.read_csv('/kaggle/input/homework-for-students3/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/homework-for-students3/test.csv', index_col=0)



sample_submission = pd.read_csv('/kaggle/input/homework-for-students3/sample_submission.csv')

free_zipcde_database = pd.read_csv('/kaggle/input/homework-for-students3/free-zipcode-database.csv')

statelatlong = pd.read_csv('/kaggle/input/homework-for-students3/statelatlong.csv')
df_train.shape, df_test.shape
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
df_train.groupby('issue_d').agg(np.sum)
df_train['issue_d2'] = df_train['issue_d']

df_test['issue_d2'] = df_test['issue_d']
df_train['issue_d2'] = df_train['issue_d2'].replace('Jan-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Feb-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Mar-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Apr-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('May-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Jun-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Jul-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Aug-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Sep-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Oct-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Nov-', '', regex=True)

df_train['issue_d2'] = df_train['issue_d2'].replace('Dec-', '', regex=True)
df_train.groupby('issue_d2').agg(np.sum)
df_train.drop(df_train.index[df_train['issue_d2'] == '2007'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2008'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2009'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2010'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2011'], inplace=True)

### added Thank you, KK, for giving me great suggestion. Your suggestion made my score improved a lot! 

df_train.drop(df_train.index[df_train['issue_d2'] == '2012'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2013'], inplace=True)

df_train.drop(df_train.index[df_train['issue_d2'] == '2014'], inplace=True)
df_train.describe()
df_test.describe()
df_train['loan_condition'].isnull().sum()
df_train['loan_condition'].value_counts()
## X_train と X_test の分割は、先に表結合を済ませた方が無難。ターゲットエンコーディングではまった。
# df = pd.DataFrame(df_train['purpose'])

# df_edit = df.copy()

# df_edit = pd.DataFrame(df_test['purpose'])

# df_diff = pd.concat([df,df_edit])

# df_diff = df_diff.drop_duplicates(keep=False)

# # keep="last"でdf_edit側の値を残す

# df_diff.drop_duplicates(subset="purpose",keep="last")
# df_train[df_train['purpose'].str.contains('educational', regex=True)]
# df_test[df_test['purpose'].str.contains('educational', regex=True)]
# #newdf = df[df['C'] != 'XYZ']

# df_train = df_train[df_train['purpose'] != 'educational']
# df = pd.DataFrame(df_train['addr_state'])

# df_edit = df.copy()

# df_edit = pd.DataFrame(df_test['addr_state'])

# df_diff = pd.concat([df,df_edit])

# df_diff = df_diff.drop_duplicates(keep=False)

# # keep="last"でdf_edit側の値を残す

# df_diff.drop_duplicates(subset="addr_state",keep="last")
# df_train[df_train['addr_state'].str.contains('ID', regex=True)]
# df_test[df_test['addr_state'].str.contains('ID', regex=True)]
# #newdf = df[df['C'] != 'XYZ']

# df_test = df_test[df_test['addr_state'] != 'ID']
# df_test[df_test['addr_state'].str.contains('ID', regex=True)]
# #C:\Users\hiroa>cat kadai_zipcode.txt | perl -lane "print \"df_test[df_test['zip_code'].str.contains('$_', regex=True)]\""

# df_test[df_test['zip_code'].str.contains('819xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('509xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('503xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('522xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('269xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('892xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('929xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('205xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('709xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('849xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('694xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('520xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('817xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('568xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('399xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('649xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('862xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('507xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('202xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('621xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('552xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('909xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('698xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('055xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('515xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('528xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('966xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('521xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('353xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('009xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('872xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('987xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('525xx', regex=True)]
# df_test[df_test['zip_code'].str.contains('525xx', regex=True)]
# df_train[df_train['zip_code'].str.contains('525xx', regex=True)]
# #newdf = df[df['C'] != 'XYZ']

# df_test = df_test[df_test['zip_code'] != '525xx']
# df_test = df_test[df_test['zip_code'] != '819xx']

# df_test = df_test[df_test['zip_code'] != '509xx']

# df_test = df_test[df_test['zip_code'] != '503xx']

# df_test = df_test[df_test['zip_code'] != '522xx']

# df_test = df_test[df_test['zip_code'] != '269xx']

# df_test = df_test[df_test['zip_code'] != '892xx']

# df_test = df_test[df_test['zip_code'] != '929xx']

# df_test = df_test[df_test['zip_code'] != '205xx']

# df_test = df_test[df_test['zip_code'] != '709xx']

# df_test = df_test[df_test['zip_code'] != '849xx']

# df_test = df_test[df_test['zip_code'] != '694xx']

# df_test = df_test[df_test['zip_code'] != '520xx']

# df_test = df_test[df_test['zip_code'] != '817xx']

# df_test = df_test[df_test['zip_code'] != '568xx']

# df_test = df_test[df_test['zip_code'] != '399xx']

# df_test = df_test[df_test['zip_code'] != '649xx']

# df_test = df_test[df_test['zip_code'] != '862xx']

# df_test = df_test[df_test['zip_code'] != '507xx']

# df_test = df_test[df_test['zip_code'] != '202xx']

# df_test = df_test[df_test['zip_code'] != '621xx']

# df_test = df_test[df_test['zip_code'] != '552xx']

# df_test = df_test[df_test['zip_code'] != '909xx']

# df_test = df_test[df_test['zip_code'] != '698xx']

# df_test = df_test[df_test['zip_code'] != '055xx']

# df_test = df_test[df_test['zip_code'] != '515xx']

# df_test = df_test[df_test['zip_code'] != '528xx']

# df_test = df_test[df_test['zip_code'] != '966xx']

# df_test = df_test[df_test['zip_code'] != '521xx']

# df_test = df_test[df_test['zip_code'] != '353xx']

# df_test = df_test[df_test['zip_code'] != '009xx']

# df_test = df_test[df_test['zip_code'] != '872xx']

# df_test = df_test[df_test['zip_code'] != '987xx']

# # #C:\Users\hiroa>cat kadai_zipcode.txt | perl -lane "print \"df_test[df_test['zip_code'].str.contains('$_', regex=True)]\""

# df_test[df_test['zip_code'].str.contains('819xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('509xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('503xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('522xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('269xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('892xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('929xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('205xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('709xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('849xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('694xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('520xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('817xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('568xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('399xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('649xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('862xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('507xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('202xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('621xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('552xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('909xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('698xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('055xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('515xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('528xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('966xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('521xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('353xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('009xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('872xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('987xx', regex=True)]

# df_test[df_test['zip_code'].str.contains('525xx', regex=True)]
y_train = df_train['loan_condition'].copy()

X_train = df_train.drop(['loan_condition'], axis=1).copy()

X_test = df_test.copy()
cats = []

others = []

for col in X_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

    else:

        others.append(col)
cats
others
for i in X_test.columns:

    print(i, X_test[i].nunique(), X_test[i].dtype)


# X_train['revol_bal'] = X_train['revol_bal'].apply(np.log1p)

# X_test['revol_bal'] = X_test['revol_bal'].apply(np.log1p)



# X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

# X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)



# X_train['tot_coll_amt'] = X_train['tot_coll_amt'].apply(np.log1p)

# X_test['tot_coll_amt'] = X_test['tot_coll_amt'].apply(np.log1p)



# X_train['tot_cur_bal'] = X_train['tot_cur_bal'].apply(np.log1p)

# X_test['tot_cur_bal'] = X_test['tot_cur_bal'].apply(np.log1p)
others = ['loan_amnt',

 'installment',

 'annual_inc',

 'dti',

 'delinq_2yrs',

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

 'acc_now_delinq',

 'tot_coll_amt',

 'tot_cur_bal']
# Clipping

for col in others:

    p01 = X_train[col].quantile(0.01)

    p99 = X_train[col].quantile(0.99)

    X_train[col] = X_train[col].clip(p01, p99)

    

    p01 = X_test[col].quantile(0.01)

    p99 = X_test[col].quantile(0.99)

    X_test[col] = X_test[col].clip(p01, p99)
#数値変換することで精度が上がる特徴量を変換

X_train['emp_length'] = X_train['emp_length'].replace('years', '', regex=True)

X_train['emp_length'] = X_train['emp_length'].replace('year', '', regex=True)

X_train['emp_length'] = X_train['emp_length'].replace("\+", '', regex=True)

X_train['emp_length'] = X_train['emp_length'].replace("\<", '', regex=True)

X_train['emp_length'] = X_train['emp_length'].replace(" ", '', regex=True)

X_train['emp_length'] = X_train['emp_length'].astype("float32")



X_test['emp_length'] = X_test['emp_length'].replace('years', '', regex=True)

X_test['emp_length'] = X_test['emp_length'].replace('year', '', regex=True)

X_test['emp_length'] = X_test['emp_length'].replace("\+", '', regex=True)

X_test['emp_length'] = X_test['emp_length'].replace("\<", '', regex=True)

X_test['emp_length'] = X_test['emp_length'].replace(" ", '', regex=True)

X_test['emp_length'] = X_test['emp_length'].astype("float32")
X_train['emp_length'].head(100)
print(X_train['emp_length'].dtype)

print(X_test['emp_length'].dtype)
print(X_train['emp_length'].head())
# mapping。Tree 系は、数値の順序性が重要

def mapping(map_col, mapping):

    X_train[map_col] = X_train[map_col].map(mapping)

    X_test[map_col] = X_test[map_col].map(mapping)
grade_mapping = { "A": 1,"B": 2,"C": 3,"D": 4,"E": 5,"F": 6,"G": 7 }



subgrade_mapping = {"A1": 1,"A2": 2,"A3": 3,"A4": 4,"A5": 5,"B1": 6,"B2": 7,"B3": 8,"B4": 9,"B5": 10,

                    "C1": 11,"C2": 12,"C3": 13,"C4": 14,"C5": 15,"D1": 16,"D2": 17,"D3": 18,"D4": 19,"D5": 20,

                    "E1": 21,"E2": 22,"E3": 23,"E4": 24,"E5": 25,"F1": 26,"F2": 27,"F3": 28,"F4": 29,"F5": 30,

                    "G1": 31,"G2": 32,"G3": 33,"G4": 34,"G5": 35

                   }
mapping('grade', grade_mapping)

mapping('sub_grade', subgrade_mapping)
#後で四則演算するので、型はint のままでおいておく

X_train['grade+subg'] = X_train['grade'] + X_train['sub_grade']

#X_train['grade*subg'] = X_train['grade'] * X_train['sub_grade']

#X_train['grade_subg'] = X_train['sub_grade'] - X_train['grade']

#X_train['grade%subg'] = X_train['sub_grade'] % X_train['grade']
X_train['grade'] = X_train['grade'].astype("object")

X_test['grade'] = X_test['grade'].astype("object")



X_train['sub_grade'] = X_train['sub_grade'].astype("object")

X_test['sub_grade'] = X_test['sub_grade'].astype("object")
X_train = X_train.drop(['purpose'], axis=1)

X_train = X_train.drop(['earliest_cr_line'], axis=1)

X_train = X_train.drop(['issue_d'], axis=1)

X_train = X_train.drop(['issue_d2'], axis=1)



X_test = X_test.drop(['purpose'], axis=1)

X_test = X_test.drop(['earliest_cr_line'], axis=1)

X_test = X_test.drop(['issue_d'], axis=1)

X_test = X_test.drop(['issue_d2'], axis=1)
# X_train

#X_train['tot_coll_amt%_annual_inc'] = X_train['tot_coll_amt'] / X_train['annual_inc']

X_train['tot_coll_amt*_annual_inc'] = X_train['tot_coll_amt'] * X_train['annual_inc']



#X_train['tot_cur_bal%_annual_inc'] = X_train['tot_cur_bal'] / X_train['annual_inc']

X_train['tot_cur_bal*_annual_inc'] = X_train['tot_cur_bal'] * X_train['annual_inc']



X_train['tot_coll_amt%_loan_amnt'] = X_train['tot_coll_amt'] / X_train['loan_amnt']

X_train['tot_coll_amt*_loan_amnt'] = X_train['tot_coll_amt'] * X_train['loan_amnt']



X_train['tot_cur_bal%_loan_amnt'] = X_train['tot_cur_bal'] / X_train['loan_amnt']

X_train['tot_cur_bal*_loan_amnt'] = X_train['tot_cur_bal'] * X_train['loan_amnt']



#X_train['loan_amnt_%_annual_inc'] = X_train['loan_amnt'] / X_train['annual_inc']

X_train['loan_amnt_*_annual_inc'] = X_train['loan_amnt'] * X_train['annual_inc']



X_train['dti_%_annual_inc'] = X_train['dti'] / X_train['annual_inc']

X_train['dti_*_annual_inc'] = X_train['dti'] * X_train['annual_inc']



#X_test['tot_coll_amt%_annual_inc'] = X_test['tot_coll_amt'] / X_test['annual_inc']

X_test['tot_coll_amt*_annual_inc'] = X_test['tot_coll_amt'] * X_test['annual_inc']



#X_test['tot_cur_bal%_annual_inc'] = X_test['tot_cur_bal'] / X_test['annual_inc']

X_test['tot_cur_bal*_annual_inc'] = X_test['tot_cur_bal'] * X_test['annual_inc']



X_test['tot_coll_amt%_loan_amnt'] = X_test['tot_coll_amt'] / X_test['loan_amnt']

X_test['tot_coll_amt*_loan_amnt'] = X_test['tot_coll_amt'] * X_test['loan_amnt']



X_test['tot_cur_bal%_loan_amnt'] = X_test['tot_cur_bal'] / X_test['loan_amnt']

X_test['tot_cur_bal*_loan_amnt'] = X_test['tot_cur_bal'] * X_test['loan_amnt']



#X_test['loan_amnt_%_annual_inc'] = X_test['loan_amnt'] / X_test['annual_inc']

X_test['loan_amnt_*_annual_inc'] = X_test['loan_amnt'] * X_test['annual_inc']



X_test['dti_%_annual_inc'] = X_test['dti'] / X_test['annual_inc']

X_test['dti_*_annual_inc'] = X_test['dti'] * X_test['annual_inc']



X_test['grade+subg'] = X_test['grade'] + X_test['sub_grade']

#X_test['grade*subg'] = X_test['grade'] * X_test['sub_grade']

# X_test['grade_subg'] = X_test['sub_grade'] - X_test['grade']

# X_test['grade%subg'] = X_test['sub_grade'] % X_test['grade']
#nullフラグ

X_train['emp_length_nullflg'] = X_train['emp_length'].apply(lambda x: 0 if not pd.isnull(x) else 1)

X_train['emp_title_nullflg'] = X_train['emp_title'].apply(lambda x: 0 if not pd.isnull(x) else 1)

X_train['title_nullflg'] = X_train['title'].apply(lambda x: 0 if not pd.isnull(x) else 1)



X_test['emp_length_nullflg'] = X_test['emp_length'].apply(lambda x: 0 if not pd.isnull(x) else 1)

X_test['emp_title_nullflg'] = X_test['emp_title'].apply(lambda x: 0 if not pd.isnull(x) else 1)

X_test['title_nullflg'] = X_test['title'].apply(lambda x: 0 if not pd.isnull(x) else 1)

# #効きそうなflg

# X_train['debt_consolidation_flg'] = X_train['title'].apply(lambda x: 1 if x == 'debt_consolidation' else '0')

# X_test['debt_consolidation_flg'] = X_test['title'].apply(lambda x: 1 if x == 'debt_consolidation' else '0')
# zip_code を1文字（州の意）に変換（鎌田さんありがとうございます）
X_train['zip_code'] = X_train['zip_code'].str[:1]

X_test['zip_code'] = X_test['zip_code'].str[:1]

X_train['zip_code'] = X_train['zip_code'].astype("object")

X_test['zip_code'] = X_test['zip_code'].astype("object")
X_train['zip_code']
X_train['emp_title'] = X_train['emp_title'].str[:4]

X_test['emp_title'] = X_test['emp_title'].str[:4]
X_train['emp_title'].value_counts().head(100)
X_train['emp_title'] = X_train['emp_title'].astype("object")

X_test['emp_title'] = X_test['emp_title'].astype("object")
X_train_na = (X_train.isnull().sum() / len(X_train)) * 100

X_train_na = X_train_na.drop(X_train_na[X_train_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :X_train_na})

missing_data.head(50)
X_test_na = (X_test.isnull().sum() / len(X_test)) * 100

X_test_na = X_test_na.drop(X_test_na[X_test_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :X_test_na})

missing_data.head(50)
X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_train.median(), inplace=True)
X_train_na = (X_train.isnull().sum() / len(X_train)) * 100

X_train_na = X_train_na.drop(X_train_na[X_train_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :X_train_na})

missing_data.head(50)
X_train["emp_title"] = X_train["emp_title"].fillna("None")

X_test["emp_title"] = X_test["emp_title"].fillna("None")



X_train["title"] = X_train["title"].fillna("None")

X_test["title"] = X_test["title"].fillna("None")
X_test_na = (X_test.isnull().sum() / len(X_test)) * 100

X_test_na = X_test_na.drop(X_test_na[X_test_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :X_test_na})

missing_data.head(50)
cats = []

others = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

    else:

        others.append(col)
cats
oe_cols = ['emp_title',

 'home_ownership',

 'title',

 'zip_code',

 'addr_state',

 'initial_list_status',

 'application_type']
oe = OrdinalEncoder(cols=oe_cols, return_df=False)



X_train[oe_cols] = oe.fit_transform(X_train[oe_cols])

X_test[oe_cols] = oe.transform(X_test[oe_cols])
X_train
X_test
X_train.describe()
X_test.describe()
'''

#emp_title の中に、trainになくて、test にあるデータがある

#cols = ['grade', 'sub_grade','emp_title','home_ownership','title','zip_code','addr_state','initial_list_status','application_type','emp_length_nullflg', 'emp_title_nullflg', 'title_nullflg']

cols = ['grade', 'sub_grade','home_ownership','title','zip_code','addr_state','initial_list_status','application_type']



target = 'loan_condition'



for col in cols:

    X_temp = pd.concat([X_train, y_train], axis=1)

    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary) 



    

    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

    

    X_train[col] = enc_train

    X_test[col] = enc_test

'''
#https://qiita.com/rshinji/items/80e844beab57c9726b12

class TargetEncoding_ws(object):

    """

    DFと変換したいカラムリスト、targetを引数として、Target Encoding with Smoothingを行う

    引数

    dataframe : DF全体 (pd.DataFrame)

    target : 目的変数のカラム (np.ndarray or np.Series)

    list_cols : 変換したいカラムリスト (list[str])

    k : smoothingのハイパーパラメータ (int)

    impute : 未知のカテゴリに平均を入れるか (boolean)

    """

    def __init__(self, list_cols, k=100, impute=True):

        self.df = None

        self.target = None

        self.list_cols = list_cols

        self.k = k

        self.impute = impute

        self.target_map = {}

        self.target_mean = None



    def sigmoid(self, x, k):

        return 1 / (1 + np.exp(- x / k))



    def fit_univariate(self, target, col):

        """

        一つの変数に対するTarget_Encoding

        col : TargetEncodingしたい変数名

        """

        df = self.df.copy()

        k = self.k

        df["target"] = target

        n_i = df.groupby(col).count()["target"]



        lambda_n_i = self.sigmoid(n_i, k)

        uni_map = df.groupby(col).mean()["target"]



        return lambda_n_i * df.loc[:, "target"].mean() + (1 - lambda_n_i) * uni_map



    def fit(self, data, target):

        """

        複数カラムにも対応したTargetEncoding

        """

        self.df = data.copy()

        self.target = target



        if self.impute == True:

            self.target_mean = target.mean()



        #各カラムのmapを保存

        for col in list_cols:

            self.target_map[col] = self.fit_univariate(target, col)



    def transform(self, x):

        list_cols = self.list_cols

        x_d = x.copy()

        for col in list_cols:

            x_d.loc[:, col] = x_d.loc[:, col].map(self.target_map[col])



            #impute

            if self.impute == True:

                x_d.loc[:, col] = np.where(x_d.loc[:, col].isnull(), self.target_mean, x_d.loc[:, col])



        return x_d
cats
list_cols =[

 'emp_title',

 'home_ownership',

 'title',

 'zip_code',

 'addr_state',

 'initial_list_status',

 'application_type']



te = TargetEncoding_ws(list_cols=list_cols, k=200, impute=True)

te.fit(X_train, y_train)



X_train = te.transform(X_train)

X_test = te.transform(X_test)



display(te.transform(X_train).head())

display(te.transform(X_test).head())
X_train.describe()
X_test.describe()
# X_train.fillna(-9999, inplace=True)

# X_test.fillna(-9999, inplace=True)
import seaborn as sns

X_train_corr = X_train.corr()

print(X_train_corr)

sns.heatmap(X_train_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
import seaborn as sns

X_test_corr = X_test.corr()

print(X_train_corr)

sns.heatmap(X_train_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
for i in X_train.columns:

    print(i, X_train[i].nunique(), X_train[i].dtype)
from pylab import rcParams

rcParams['figure.figsize'] = 20,20 # グラフのサイズを大きくする

X_train.hist(bins=20);

plt.tight_layout() # グラフ同士が重ならないようにする

plt.show()
from pylab import rcParams

rcParams['figure.figsize'] = 20,20 # グラフのサイズを大きくする

X_test.hist(bins=20);

plt.tight_layout() # グラフ同士が重ならないようにする

plt.show()


# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。⇒yes

scores = []

scores2 = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #

    #clf = GradientBoostingClassifier(n_estimators=85, 

    #                                learning_rate=0.25,

    #                                #max_depth=2, #added at version 2

    #                                random_state=1)

    #

    lgbmc = LGBMClassifier(objective='binary',

                         n_estimators=10000, 

                         class_weight = 'balanced',

                         #learning_rate=0.3, 

                         #max_depth=2,

                         #subsample= 0.8,

                         #bagging_fraction=0.7,

                         #feature_fraction=0.7,

                         importance_type='gain', 

                         random_state=71, 

                         silent = 1,

                         n_jobs=-1)

    #bagging_fraction': 0.7, 'feature_fraction': 0.7, 'learning_rate': 0.20000000000000004, 'max_depth': 2} 0.8161721235129574

    

    #clf.fit(X_train_, y_train_)

    lgbmc.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='binary_logloss', verbose=100, eval_set=[(X_val, y_val)])

    # verbose_eval=False

    

    y_pred = lgbmc.predict_proba(X_val)[:,1]

    score2 = roc_auc_score(y_train_, lgbmc.predict_proba(X_train_)[:,1])

    scores2.append(score2)

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)



    print('===================================================')

    print('train CV Score of Fold_%d is %f' % (i, score2))

    print('---------------------------------------------------')

    print('val CV Score of Fold_%d is %f' % (i, score))

    print('---------------------------------------------------')

    print('diff', score - score2)

    print('===================================================')

# from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold



# # CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。→train にはなくて、testにはIDのお客さんが存在する

# scores = []

# scores2 = []



# groups = X_train.addr_state.values

# gkf = GroupKFold(n_splits=5)

# #skf = StratifiedKFold(n_splits=3, random_state=71, shuffle=True)



# for i, (train_ix, test_ix) in tqdm(enumerate(gkf.split(X_train, y_train, groups))):

#     X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

#     X_val, y_val, group_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

    

#     '''

#     clf = GradientBoostingClassifier(n_estimators=85, 

#                                     learning_rate=0.25,

#                                     #max_depth=2, #added at version 2

#                                     random_state=1)

#     '''

#     lgbmc = LGBMClassifier(objective='binary',

#                          n_estimators=10000, 

#                          boosting_type='gbdt',

#                          class_weight = 'balanced',

#                          #learning_rate=0.3, 

#                          #subsample= 0.8,

#                          #bagging_fraction=0.7,

#                          #feature_fraction=0.7,

#                          importance_type='gain', 

#                          random_state=71, 

#                          silent = 1,

#                          n_jobs=-1)

    

#     #bagging_fraction': 0.7, 'feature_fraction': 0.7, 'learning_rate': 0.20000000000000004, 'max_depth': 2} 0.8161721235129574

    

#     #clf.fit(X_train_, y_train_)

#     lgbmc.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='binary_logloss', verbose=200, eval_set=[(X_val, y_val)])

#     # verbose_eval=False

    

#     y_pred = lgbmc.predict_proba(X_val)[:,1]

#     score2 = roc_auc_score(y_train_, lgbmc.predict_proba(X_train_)[:,1])

#     scores2.append(score2)

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)



#     print('===================================================')

#     print('train CV Score of Fold_%d is %f' % (i, score2))

#     print('---------------------------------------------------')

#     print('val CV Score of Fold_%d is %f' % (i, score))

#     print('---------------------------------------------------')

#     print('diff', score - score2)

#     print('===================================================')

print('val avg:', np.mean(scores))

print(scores)

print('=============================')

print('train avg:', np.mean(scores2))

print(scores2)

print('=============================')

print('diff',np.mean(scores) - np.mean(scores2))
# Plot feature importance

#importances = clf.feature_importance(importance_type='gain')



# Initialize an empty array to hold feature importances

feature_importances = np.zeros(X_train.shape[1])



importances = lgbmc.feature_importances_



indices = np.argsort(importances)[::-1]



feat_labels = X_train.columns[0:]

for f in range(X_train.shape[1]):

    IMPORTANCES_LIST = pd.DataFrame([["%2d) %-*s %f" % (f +1, 30, feat_labels[indices[f]], importances[indices[f]])]] )

#    IMPORTANCES_LIST.to_csv('./data/feature_importances.csv', mode='a', header=False, index=False)

    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
fig, ax = plt.subplots(figsize=(20, 10))

lgb.plot_importance(lgbmc, max_num_features=50, ax=ax, importance_type='gain')


from hyperopt import fmin, tpe, hp, rand, Trials



def objective(space):

    hscores = []



    skf = StratifiedKFold(n_splits=3, random_state=71, shuffle=True)



    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

        

        clf = LGBMClassifier(n_estimators=9999, random_state=71, class_weight = 'balanced', importance_type='gain', **space) 



        clf.fit(X_train_, y_train_, early_stopping_rounds=10, eval_metric='auc', verbose=200, eval_set=[(X_val, y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        hscores.append(score)

        

    hscores = np.array(hscores)

    print(hscores.mean())

    



    return -hscores.mean()



space ={

        #'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),

        'num_leaves': hp.choice('num_leaves', np.linspace(10, 200, 50, dtype=int)),

        'subsample': hp.uniform ('subsample', 0.8, 1),

        'learning_rate' : hp.quniform('learning_rate', 0.1, 0.5, 0.05),

        'colsample_bytree' : hp.quniform('colsample_bytree', 0.7, 1, 0.05)

    }



trials = Trials()



best = fmin(fn=objective,

              space=space, 

              algo=tpe.suggest,

              max_evals=20, 

              trials=trials, 

              rstate=np.random.RandomState(71) 

             )

best_model = LGBMClassifier(**best)
trials.best_trial['result']
ho_best = - pd.Series(trials.losses()).min()
ho_best
print(best)

#{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'num_leaves': 10, 'subsample': 0.9929417385040324} 

#⇒num_leaves が10になるのは過学習のせいか
# # 全データで再学習し、testに対して予測する

# lgbmc.fit(X_train, y_train)



# y_pred = lgbmc.predict_proba(X_test)[:,1]

if ho_best <= np.mean(scores):

    # 全データで再学習し、testに対して予測する

    lgbmc.fit(X_train, y_train)

    y_pred = lgbmc.predict_proba(X_test)[:,1]

else:

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict_proba(X_test)[:,1]
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('/kaggle/input/homework-for-students3/sample_submission.csv')

submission.loan_condition = y_pred

submission.to_csv('./submission.csv', index=False)
submission.head()