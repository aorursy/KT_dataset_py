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

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



pd.set_option("display.max_columns",2000)


path_to_train = '/kaggle/input/homework-for-students3/train.csv'

path_to_test = '/kaggle/input/homework-for-students3/test.csv'

path_to_submission = '/kaggle/input/homework-for-students3/sample_submission.csv'

"""

path_to_train = './train.csv'

path_to_test = './test.csv'

path_to_submission = './sample_submission.csv'

"""
# データ読み込み

df_train = pd.read_csv(path_to_train, index_col=0)

df_test = pd.read_csv(path_to_test, index_col=0)
#行数と項目数の確認

df_train.shape, df_test.shape
# 基本統計量の確認

df_train.describe()
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。



df = pd.DataFrame()

for col in df_train.columns:

    print("%-30s%-15s%s"%(col,df_train[col].dtype,df_train[col].nunique()))

    #if X_train[col].dtype == 'object':
###データ範囲を絞る　issue_d 2015年分のみ

df_train = df_train[df_train["issue_d"].str.contains("-201[45]")]

print(df_train["issue_d"].value_counts())

df_train["issue_d"].value_counts().hist()
df_test["issue_d"].value_counts()
###カラム削除

drop_cols = ["issue_d","earliest_cr_line"]



#df_train = df_train.drop(drop_cols,axis=1)

#df_test = df_test.drop(drop_cols,axis=1)

df_test.isnull().sum()
### X,Y分離

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test



### テキスト用にコピー

#TXT_train = X_train.emp_title.copy()

#TXT_test = X_test.emp_title.copy()
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        print(X_train[col].value_counts())

        

        print(col, X_train[col].nunique())
### 欠損処理

nodata_cols = ["title","application_type"]

median_cols = ["delinq_2yrs","inq_last_6mths","mths_since_last_delinq","mths_since_last_record"

               ,"open_acc","pub_rec","collections_12_mths_ex_med","mths_since_last_major_derog"

               ,"acc_now_delinq","tot_coll_amt","tot_cur_bal","dti","revol_util","total_acc"]

mode_cols   = ["emp_length"]



## カテゴリ系

X_train[nodata_cols]  = X_train[nodata_cols].fillna("nodata")

X_test[nodata_cols]   = X_test[nodata_cols].fillna("nodata")



##　数値系（中央値）

X_train[median_cols]  = X_train[median_cols].fillna(df_train[median_cols].median())

X_test[median_cols]   = X_test[median_cols].fillna(df_test[median_cols].median())



##　最頻値

#X_train["emp_length"] = X_train["emp_length"].fillna(df_train["emp_length"].mode()[0])

#X_test["emp_length"]  = X_test["emp_length"].fillna(df_test["emp_length"].mode()[0])

X_train["emp_length"]  = X_train["emp_length"].fillna(0)

X_test["emp_length"]   = X_test["emp_length"].fillna(0)



### 確認

print(X_test.isnull().sum())
###欠損フラグ

havenullcol = X_train.columns[X_train.isnull().sum()!=0].values 

print(havenullcol)

for col in havenullcol:

    X_train[col+'null'] = 0

    X_train[col+'null'][X_train[col].isnull()] = 1

    X_test[col+'null'] = 0 

    X_test[col+'null'][X_test[col].isnull()] = 1

    

print(X_train)

    
#対数化

'''

std_cols = ["annual_inc","open_acc","total_acc"]

print(X_train.isnull().sum())

plt.subplot(1,2,1)

X_train[std_cols].hist()



X_train[std_cols] =X_train[std_cols].apply(np.log1p)

X_test[std_cols] =X_test[std_cols].apply(np.log1p)



plt.subplot(1,2,2)

X_train[std_cols].hist()

print(X_train.isnull().sum())

'''
# 外れ値の除外

num_cols = ["loan_amnt","installment","annual_inc","dti",

            "delinq_2yrs","inq_last_6mths","mths_since_last_delinq",

            "mths_since_last_record","open_acc","pub_rec","revol_bal",

            "revol_util","total_acc","collections_12_mths_ex_med",

            "mths_since_last_major_derog","acc_now_delinq","tot_coll_amt","tot_cur_bal"]
#　スケーリング

print(X_train.isnull().sum())

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])



scaler.fit(X_test[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])

#カテゴリ（OrdinalEncoding）

mappingdict = { "grade": { "A": 1,"B": 2,"C": 3,"D": 4,"E": 5,"F": 6,"G": 7 },

     "sub_grade": {"A1": 1,"A2": 2,"A3": 3,"A4": 4,"A5": 5,

    "B1": 6,"B2": 7,"B3": 8,"B4": 9,"B5": 10,

    "C1": 11,"C2": 12,"C3": 13,"C4": 14,"C5": 15,

    "D1": 16,"D2": 17,"D3": 18,"D4": 19,"D5": 20,

    "E1": 21,"E2": 22,"E3": 23,"E4": 24,"E5": 25,

    "F1": 26,"F2": 27,"F3": 28,"F4": 29,"F5": 30,

    "G1": 31,"G2": 32,"G3": 33,"G4": 34,"G5": 35

    }

    ,"emp_length": { "10+ years": 10,"9 years": 9,"8 years": 8,"7 years": 7, "6 years": 6,"5 years": 5,"4 years": 4,"3 years": 3, "2 years": 2,"1 year": 1,"< 1 year": 0,"n/a": "" }

    ,"addr_state": {"CA":51,

"TX":50,

"NY":49,

"FL":48,

"IL":47,

"PA":46,

"OH":45,

"NJ":44,

"GA":43,

"NC":42,

"MA":41,

"VA":40,

"MI":39,

"WA":38,

"MD":37,

"IN":36,

"MN":35,

"TN":34,

"CO":33,

"WI":32,

"MO":31,

"AZ":30,

"CT":29,

"LA":28,

"OR":27,

"SC":26,

"AL":25,

"KY":24,

"OK":23,

"IA":22,

"KS":21,

"UT":20,

"NV":19,

"DC":18,

"AR":17,

"NE":16,

"MS":15,

"NM":14,

"HI":13,

"NH":12,

"WV":11,

"DE":10,

"ID":9,

"ME":8,

"ND":7,

"RI":6,

"AK":5,

"SD":4,

"MT":3,

"WY":2,

"VT":1}} 

X_train = X_train.replace(mappingdict)

X_test = X_test.replace(mappingdict)

mappingcol = ['grade','sub_grade','addr_state','emp_length']

X_train[mappingcol] = X_train[mappingcol].astype(float)

X_test[mappingcol] = X_test[mappingcol].astype(float)



X_train.columns
###収入に対するローンの割合

#loan_amnt / annual_inc

X_train['loan_to_income_ratio']=X_train["loan_amnt"]/(X_train["annual_inc"]+1)

X_test['loan_to_income_ratio']=X_test["loan_amnt"]/(X_train["annual_inc"]+1)



#X_train['loan_to_income_ratio_flg'] = X_train['loan_to_income_ratio'].map(lambda x: 1if x > 0.25 else 0)

#X_test['loan_to_income_ratio_flg'] = X_test['loan_to_income_ratio'].map(lambda x: 1 if x > 0.25 else 0)



###収入に対する返済額の割合

#installment / annual_inc

X_train['installment_to_income_ratio']=X_train["installment"]/(X_train["annual_inc"]+1)

X_test['installment_to_income_ratio']=X_test["installment"]/(X_train["annual_inc"]+1)

###収入に対するリボ残高の割合

#revol_bal / annual_inc

X_train['revol_bal_to_income_ratio']=X_train["revol_bal"]/(X_train["annual_inc"]+1)

X_test['revol_bal_to_income_ratio']=X_test["revol_bal"]/(X_train["annual_inc"]+1)

###収入に対する徴収総額の割合

#revol_bal / annual_inc

X_train['tot_coll_amt_to_income_ratio']=X_train["tot_coll_amt"]/(X_train["annual_inc"]+1)

X_test['tot_coll_amt_to_income_ratio']=X_test["tot_coll_amt"]/(X_train["annual_inc"]+1)





###全口座残高に対する徴収総額の割合

#revol_bal / tot_cur_bal

X_train['tot_coll_amt_to_tot_cur_bal_ratio']=X_train["tot_coll_amt"]/(X_train["tot_cur_bal"]+1)

X_test['tot_coll_amt_to_tot_cur_bal_ratio']=X_test["tot_coll_amt"]/(X_train["tot_cur_bal"]+1)

###全口座残高に対する返済額の割合

#revol_bal / tot_cur_bal

X_train['installment_to_tot_cur_bal_ratio']=X_train["installment"]/(X_train["tot_cur_bal"]+1)

X_test['installment_to_tot_cur_bal_ratio']=X_test["installment"]/(X_train["tot_cur_bal"]+1)





###grade

X_train['grade_by_subgrade']=X_train["grade"]*X_train["sub_grade"]

X_test['grade_by_subgrade']=X_test["grade"]*X_test["sub_grade"]



X_train['grade_by_annual_inc']=X_train["grade"]*X_train["annual_inc"]

X_test['grade_by_annual_inc']=X_test["grade"]*X_test["annual_inc"]



X_train['sub_grade_by_annual_inc']=X_train["sub_grade"]*X_train["annual_inc"]

X_test['sub_grade_by_annual_inc']=X_test["sub_grade"]*X_test["annual_inc"]



### purpose

#X_train['purpose_count'] = X_train['purpose'].map(X_train['purpose'].value_counts())

#X_test['purpose_count'] = X_test['purpose'].map(X_test['purpose'].value_counts())

### purpose

X_train['purpose_count'] = X_train['purpose'].map(pd.concat([X_train,X_test])['purpose'].value_counts())

X_test['purpose_count'] = X_test['purpose'].map(pd.concat([X_train,X_test])['purpose'].value_counts())



###emp_title

#X_train['emp_title_count'] = X_train['emp_title'].map(X_train['emp_title'].value_counts())

#X_test['emp_title_count'] = X_test['emp_title'].map(X_test['emp_title'].value_counts())

X_train['emp_title_count'] = X_train['emp_title'].map(pd.concat([X_train,X_test])['emp_title'].value_counts())

X_test['emp_title_count'] = X_test['emp_title'].map(pd.concat([X_train,X_test])['emp_title'].value_counts())

###　OrdinalEncode

###　dtype=objectが対象



oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

scores = []

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]

    

    clf = LGBMClassifier(boosting_type='gbdt',class_weight=None,colsample_bytree=0.8,

                            importance_type='split',learning_rate=0.05,max_depth=-1,

                            min_child_samples=20,min_child_weight=0.01,min_split_gain=0.0,

                            n_estimators=9999,n_jobs=-1,num_leaves=15,objective=None,

                            random_state=100,reg_alpha=0.0,reg_lambda=0,silent=True,

                            subsample=0.8,subsample_for_bin=200000,subsample_freq=0,verbosity=1)

 

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(y_val, y_pred))

    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく



scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

    

y_pred_test /= 5 # 最後にfold数で割

print(np.mean(scores))

print(scores)
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv(path_to_submission, index_col=0)



print(len(submission))

print(len(y_pred_test))



#submission.loan_condition = y_pred

submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')
clf.booster_.feature_importance(importance_type='gain')

imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp



fig, ax = plt.subplots(figsize=(10, 15))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain') # 変数重要をプロット