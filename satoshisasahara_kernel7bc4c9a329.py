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

import seaborn as sns

import optuna

from pandas import DataFrame, Series #pandasからDataFrameとSeriesをインポートする



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import optuna



from sklearn.feature_selection import RFE

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.ensemble import VotingClassifier

from lightgbm import LGBMClassifier

import lightgbm as lgb

import gc

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans





from hyperopt import fmin, tpe, hp, rand, Trials
#ファイル読込

df_train_ = pd.read_csv('../input/homework-for-students2/train.csv', index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv',index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])



#df_train_ = pd.read_csv('../input/homework-for-students2/train.csv', index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])

#df_test = pd.read_csv('../input/homework-for-students2/test.csv',index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])



#追加特徴量の読み込み

#df_spi = pd.read_csv('../input/homework-for-students2/spi.csv',index_col=0)

#df_zipcode = pd.read_csv('../input/homework-for-students2/free-zipcode-database.csv')

#df_statelatlong = pd.read_csv('../input/homework-for-students2/statelatlong.csv',index_col=0)

#df_US_GDP = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv',index_col=0)
#真面目にデータを確認する

#sns.countplot(x='acc_now_delinq',data=df_train_,hue='loan_condition')

#sns.countplot(x='addr_state',data=df_train_,hue='loan_condition')

#sns.countplot(x='annual_inc',data=df_train_,hue='loan_condition')
##ヒストグラム⇒連続値

# 描画する枠figを指定

#fig = plt.figure(figsize=(12, 8))

# 描画領域を1行1列に分割し、そのうちの1番目の分割領域をaxとする

#ax = fig.add_subplot(111)

# axにヒストグラムを描画

#ax.hist(x=df_train_['annual_inc']

#       ,bins=np.arange(0, 400000, 10000))

#ax.hist(x=df_train_['collections_12_mths_ex_med'])

#plt.show()
#あとで見分けるため、Trainに0をTestに1を付加した列を追加する

df_train_['Train_Test'] = 0

df_test['loan_condition'] = np.nan

df_test['Train_Test'] = 1
#精度向上のためここでtrainとテストを一旦マージする

df_train = pd.concat([df_train_,df_test],axis=0)

#issue_dに時系列データがあるので、ソートする

df_train = df_train.sort_values('issue_d', ascending=True)

#インデックスを振りなおす

df_train = df_train.reset_index(drop=True)
#nullを中央値で埋める。ただし、Nullフラグを同時に立てる（いっぱいNULLとStrのカラムは後ほど対処）

#欠損データに欠損フラグ立てて、欠損値を埋める

##trainデータ

df_train['missing_value_flg'] = (df_train.emp_title.isnull().astype(int).astype(str) 

                                 + df_train.emp_length.isnull().astype(int).astype(str)

                                 + df_train.annual_inc.isnull().astype(int).astype(str)

                                 + df_train.title.isnull().astype(int).astype(str)

                                 + df_train.dti.isnull().astype(int).astype(str)

                                 + df_train.delinq_2yrs.isnull().astype(int).astype(str)

                                 + df_train.earliest_cr_line.isnull().astype(int).astype(str)

                                 + df_train.inq_last_6mths.isnull().astype(int).astype(str)

                                 + df_train.mths_since_last_delinq.isnull().astype(int).astype(str)

                                 + df_train.mths_since_last_record.isnull().astype(int).astype(str)

                                 + df_train.open_acc.isnull().astype(int).astype(str)

                                 + df_train.pub_rec.isnull().astype(int).astype(str)

                                 + df_train.revol_util.isnull().astype(int).astype(str)

                                 + df_train.total_acc.isnull().astype(int).astype(str)

                                 + df_train.collections_12_mths_ex_med.isnull().astype(int).astype(str)

                                 + df_train.mths_since_last_major_derog.isnull().astype(int).astype(str)

                                 + df_train.acc_now_delinq.isnull().astype(int).astype(str)

                                 + df_train.tot_coll_amt.isnull().astype(int).astype(str)

                                 + df_train.tot_cur_bal.isnull().astype(int).astype(str))

#testデータ

'''

df_test['missing_value_flg'] = (df_test.emp_title.isnull().astype(int).astype(str) 

                                 + df_test.emp_length.isnull().astype(int).astype(str)

                                 + df_test.annual_inc.isnull().astype(int).astype(str)

                                 + df_test.title.isnull().astype(int).astype(str)

                                 + df_test.dti.isnull().astype(int).astype(str)

                                 + df_test.delinq_2yrs.isnull().astype(int).astype(str)

                                 + df_test.earliest_cr_line.isnull().astype(int).astype(str)

                                 + df_test.inq_last_6mths.isnull().astype(int).astype(str)

                                 + df_test.mths_since_last_delinq.isnull().astype(int).astype(str)

                                 + df_test.mths_since_last_record.isnull().astype(int).astype(str)

                                 + df_test.open_acc.isnull().astype(int).astype(str)

                                 + df_test.pub_rec.isnull().astype(int).astype(str)

                                 + df_test.revol_util.isnull().astype(int).astype(str)

                                 + df_test.total_acc.isnull().astype(int).astype(str)

                                 + df_test.collections_12_mths_ex_med.isnull().astype(int).astype(str)

                                 + df_test.mths_since_last_major_derog.isnull().astype(int).astype(str)

                                 + df_test.acc_now_delinq.isnull().astype(int).astype(str)

                                 + df_test.tot_coll_amt.isnull().astype(int).astype(str)

                                 + df_test.tot_cur_bal.isnull().astype(int).astype(str))

'''
#数も特徴量として使う

df_train['missing_value_sum'] = df_train.isnull().sum(axis=1)

#df_test['missing_value_sum'] = df_test.isnull().sum(axis=1)
#順序尺度特徴量をマッピングで補正する

grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}

sub_grade_mapping = {'A1': 35, 'A2': 34, 'A3': 33, 'A4': 32, 'A5': 31, 

                     'B1': 30, 'B2': 29, 'B3': 28, 'B4': 27, 'B5': 26, 

                     'C1': 25, 'C2': 24, 'C3': 23, 'C4': 22, 'C5': 21, 

                     'D1': 20, 'D2': 19, 'D3': 18, 'D4': 17, 'D5': 16, 

                     'E1': 15, 'E2': 14, 'E3': 13, 'E4': 12, 'E5': 11, 

                     'F1': 10, 'F2': 9, 'F3': 8, 'F4': 7, 'F5': 6, 

                     'G1': 5, 'G2': 4, 'G3': 3, 'G4': 2, 'G5': 1, }

emp_length_mapping = {'10+ years': 11, '9 years': 10, '8 years': 9, '7 years': 8, 

                      '6 years': 7, '5 years': 6, '4 years': 5, '3 years': 4,

                     '2 years': 3, '1 year': 2, '< 1 year': 1}

#実行

df_train['grade'] = df_train['grade'].map(grade_mapping)

df_train['sub_grade'] = df_train['sub_grade'].map(sub_grade_mapping)

df_train['emp_length'] = df_train['emp_length'].map(emp_length_mapping)



#df_test['grade'] = df_test['grade'].map(grade_mapping)

#df_test['sub_grade'] = df_test['sub_grade'].map(sub_grade_mapping)

#df_test['emp_length'] = df_test['emp_length'].map(emp_length_mapping)
num_cols = [

    'annual_inc',

    'installment',

    'dti',

    'delinq_2yrs',

    'inq_last_6mths',

    'mths_since_last_delinq',

    'mths_since_last_record',

    'open_acc',

    'pub_rec',

    'revol_util',

    'total_acc',

    'collections_12_mths_ex_med',

    'mths_since_last_major_derog',

    'acc_now_delinq',

    'tot_coll_amt',

    'tot_cur_bal',

    'emp_length'

]
#tot_cur_bal、acc_now_delinqはカテゴリと関係がありそうなのでカテゴリ毎の中央値で埋める

'''

num_cols_median = ['annual_inc','dti','delinq_2yrs','inq_last_6mths',

'mths_since_last_delinq','mths_since_last_record','open_acc',

'pub_rec','revol_util','total_acc','collections_12_mths_ex_med',

'mths_since_last_major_derog','acc_now_delinq',

'tot_coll_amt','tot_cur_bal','emp_length']

'''

#num_cols_median = ['delinq_2yrs']

for col in num_cols:

    print(col)

    #まず対象とする列のgradeごとの中央値を求める

    tmp_grouped = df_train.groupby(['grade'], as_index=False)

    target_median = tmp_grouped.median()[['grade',col]]

    target_max = tmp_grouped.max()[['grade',col]]

    target_min = tmp_grouped.min()[['grade',col]]

    target_mean = tmp_grouped.mean()[['grade',col]]

    target_std = tmp_grouped.std()[['grade',col]]

    target_25par_quant = tmp_grouped[col].quantile(0.25)

    target_75par_quant = tmp_grouped[col].quantile(0.75)



    #欠損値埋めと特徴量の派生用を同時に実行

    for row in  target_median.itertuples():

        #print(row[1],row[2],target_mean)

        #欠損値を埋める

        df_train[col].mask((df_train['grade'] == row[1]) & (df_train[col].isnull()), row[2], inplace=True)

        #df_test[col].mask((df_test['grade'] == row[1]) & (df_test[col].isnull()), row[2], inplace=True)

        #特徴量の派生

        df_train.loc[df_train['grade'] == row[1], col + '_Median'] = target_median.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_Max'] = target_max.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_Min'] = target_min.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_Mean'] = target_mean.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_Std'] = target_std.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_25Quant'] = target_25par_quant.loc[row[1]-1,col]

        df_train.loc[df_train['grade'] == row[1], col + '_75Quant'] = target_75par_quant.loc[row[1]-1,col]

        

        #df_test.loc[df_test['grade'] == row[1], col + '_Median'] = target_median.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_Max'] = target_max.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_Min'] = target_min.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_Mean'] = target_mean.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_Std'] = target_std.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_25Quant'] = target_25par_quant.loc[row[1]-1,col]

        #df_test.loc[df_test['grade'] == row[1], col + '_75Quant'] = target_75par_quant.loc[row[1]-1,col]
#Plotの結果から、mths_since_last_record、emp_length以外は99%clipする

for col in num_cols:

    upperbound, lowerbound = np.percentile(df_train[col], [1, 99])

    df_train[col] = np.clip(df_train[col], upperbound, lowerbound)
#数値系の特徴量を派生するまえに、左に山がある（右に裾が長い)データを対数変換する

#log_cols = ['dti','revol_bal','annual_inc','delinq_2yrs','inq_last_6mths' ,'open_acc']

'''

log_cols = [

'delinq_2yrs','inq_last_6mths','open_acc','pub_rec'

,'collections_12_mths_ex_med'

,'acc_now_delinq'

,'tot_coll_amt','tot_cur_bal'

]

'''

for col in num_cols:

    print(col)

    df_train[col + '_log'] = np.log1p(df_train[col])

    #df_test[col + '_log'] = np.log1p(df_test[col])
#非線形っぽい特徴を持つ特徴量をyeo-johnson変換する（対象には0値が含まれるので）⇒新規特徴量にするのはありかも。。

#やらないほうがよさそう

##pos_cols = ['dti','revol_bal','annual_inc','loan_amnt','open_acc']

#pos_cols = num_cols_median

#from sklearn.preprocessing import PowerTransformer

## 学習データに基づいて複数カラムのBox-Cox変換を定義

#pt = PowerTransformer(method='yeo-johnson')

#pt.fit(df_train[pos_cols])

## 変換後のデータで各列を置換

#df_train[pos_cols] = pt.transform(df_train[pos_cols])

#df_test[pos_cols] = pt.transform(df_test[pos_cols])
#特徴量を派生する。あとでFeature Selectionするので大量に派生する。

h = 1e-6#0割防止用

df_train['installment_divided_by_tot_cur_bal'] = df_train['installment']/(df_train['tot_cur_bal'] + h)

df_train['collections_12_mths_ex_med_times_delinq_2yrs'] = df_train['collections_12_mths_ex_med']*df_train['delinq_2yrs'] 

df_train['tot_coll_amt_divided_by_tot_cur_bal'] = df_train['tot_coll_amt']/(df_train['tot_cur_bal'] + h)

df_train['tot_coll_amt_divided_by_total_acc'] = df_train['tot_coll_amt']/(df_train['total_acc'] + h)

df_train['loan_amnt_divided_by_tot_cur_bal'] = df_train['loan_amnt']/(df_train['tot_cur_bal'] + h)

df_train['loan_amnt_divided_by_total_acc'] = df_train['loan_amnt']/(df_train['total_acc'] + h)

df_train['tot_coll_amt_divided_by_annual_inc'] = df_train['tot_coll_amt']/(df_train['annual_inc'] + h)

df_train['loan_amnt_divided_by_annual_inc'] = df_train['loan_amnt']/(df_train['annual_inc'] + h)

df_train['acc_now_delinq_divided_by_annual_inc'] = df_train['acc_now_delinq']/(df_train['annual_inc'] + h)

df_train['collections_12_mths_ex_med_divided_by_annual_inc'] = df_train['collections_12_mths_ex_med']/(df_train['annual_inc'] + h)

df_train['inq_last_6mths_divided_by_annual_inc'] = df_train['inq_last_6mths']/(df_train['annual_inc'] + h)

df_train['installment_divided_by_annual_inc'] = df_train['installment']/(df_train['annual_inc'] + h)

df_train['mths_since_last_delinq_divided_by_annual_inc'] = df_train['mths_since_last_delinq']/(df_train['annual_inc'] + h)

df_train['mths_since_last_record_divided_by_annual_inc'] = df_train['mths_since_last_record']/(df_train['annual_inc'] + h)

df_train['open_acc_divided_by_annual_inc'] = df_train['open_acc']/(df_train['annual_inc'] + h)

df_train['pub_rec_divided_by_annual_inc'] = df_train['pub_rec']/(df_train['annual_inc'] + h)

df_train['revol_util_divided_by_annual_inc'] = df_train['revol_util']/(df_train['annual_inc'] + h)

df_train['tot_cur_bal_divided_by_annual_inc'] = df_train['tot_cur_bal']/(df_train['annual_inc'] + h)

df_train['total_acc_divided_by_annual_inc'] = df_train['total_acc']/(df_train['annual_inc'] + h)
###earliest_cr_lineとissue_dが年月に分けると良いカテゴリになりそうなのでスプリットする

###⇒これはテストで必ず外挿なのでなし。⇒かわりにこの二つの差分を特徴量として再定義

'''

##earliest_cr_line

tmp_df_train = df_train.earliest_cr_line.str.split('-', expand=True)

#tmp_df_test = df_test.earliest_cr_line.str.split('-', expand=True)



tmp_df_train = tmp_df_train.rename(columns={0:'earliest_cr_line_MM',1:'earliest_cr_line_YYYY'})

#tmp_df_test = tmp_df_test.rename(columns={0:'earliest_cr_line_MM',1:'earliest_cr_line_YYYY'})



#元データとconcatして

df_train = pd.concat([df_train, tmp_df_train], axis=1)

#df_test = pd.concat([df_test, tmp_df_test], axis=1)

df_train.drop('earliest_cr_line',axis=1,inplace=True)

#df_test.drop('earliest_cr_line',axis=1,inplace=True)





##issue_d

tmp_df_train = df_train.issue_d.str.split('-', expand=True)

#tmp_df_test = df_test.issue_d.str.split('-', expand=True)



tmp_df_train = tmp_df_train.rename(columns={0:'issue_d_MM',1:'issue_d_YYYY'})

#tmp_df_test = tmp_df_test.rename(columns={0:'issue_d_MM',1:'issue_d_YYYY'})



#元データとconcatして、issue_dは消す⇒あとでやるのでここでは消さない

df_train = pd.concat([df_train, tmp_df_train], axis=1)

#df_test = pd.concat([df_test, tmp_df_test], axis=1)

'''
#earliest_cr_lineとissue_dの差をとる。口座開いてすぐ借りにくるやつはやばそう

df_train['cr_line_issue_d_diff'] = (df_train['issue_d'] - df_train['earliest_cr_line']) / np.timedelta64(1, 'D')
#カテゴリの欠損値を埋める：Zip関連削除

'''

cat_cols = ['emp_title','home_ownership','purpose','title'

            ,'addr_state','initial_list_status','application_type','zip_first_3_numbers'

            ,'missing_value_flg','ZipCodeType','City','State'

            ,'LocationType','Country','Decommisioned_y'

            ,'earliest_cr_line_MM','earliest_cr_line_YYYY'

            ,'issue_d_MM','issue_d_YYYY'

           ]

'''

cat_cols = [

    'emp_title',

    'title',

    'home_ownership',

    'purpose',

    'addr_state',

    'initial_list_status',

    'application_type',

    'zip_code',

    'missing_value_flg',

    ]    

for col in cat_cols:

    df_train[col].fillna('＃', inplace=True)

    #df_test[col].fillna('＃', inplace=True)
#件数が100件未満のカテゴリはotherとしてまとめる

for col in cat_cols:

    print(col)

    #500件未満の要素を抽出する

    under_500 = df_train[col].value_counts()

    under_500 = under_500[under_500 < 500]

    #under_500 = under_500[under_500 < 10]

    df_train.loc[df_train[col].isin(under_500.index), col] = 'Other'

    #df_test.loc[df_test[col].isin(under_500.index), col] = 'Other'
#ターゲットエンコーディングはリークの可能性が高いのでcountに戻す

cat_cols.remove('emp_title') #emp_titleはtfidfするので一旦除外する

cat_cols.remove('title') #emp_titleはtfidfするので一旦除外する



for col in cat_cols:

    print(col)

    summary = df_train[col].value_counts()

    df_train[col] = df_train[col].map(summary)
######PCAを追加して回転系（非線形の特徴量を追加する）

##PCAを実行する単位を指定する。（ビジネス的に意味がありそうな単位⇒人を表す。他には？）

##objは外す。ただし順序を持つものはOK

person_cols = [

    #'addr_state',

    'annual_inc',

    'emp_length',

    'grade',

    #'home_ownership',

    'mths_since_last_major_derog',

    'mths_since_last_record',

    'open_acc',

    'pub_rec',

    'sub_grade',

    'tot_cur_bal',

    'total_acc'

    #'zip_code'

]

loan_cols = [

    'acc_now_delinq',

    #'application_type',

    'collections_12_mths_ex_med',

    'delinq_2yrs',

    'dti',

    #'initial_list_status',

    'inq_last_6mths',

    'installment',

    'loan_amnt',

    'mths_since_last_delinq',

    #'purpose',

    'revol_bal',

    'tot_coll_amt'

]
#まずは正規化する

scaler_per = StandardScaler()

# 与えられた行列の各特徴量について､平均と標準偏差を算出

scaler_per.fit(df_train[person_cols])

# Xを標準化した行列を生成

X_std_per = scaler_per.fit_transform(df_train[person_cols])



#obj列もやる

scaler_loan = StandardScaler()

scaler_loan.fit(df_train[loan_cols])

# Xを標準化した行列を生成

X_std_loan = scaler_loan.fit_transform(df_train[loan_cols])

# PCAのインスタンスを生成し、主成分を4つまで取得

pca_per = PCA(n_components=5) 

pca_loan = PCA(n_components=5) 

X_pca_per = pca_per.fit_transform(X_std_per)

X_pca_loan = pca_loan.fit_transform(X_std_loan)
df_pca_per = pd.DataFrame(data=X_pca_per, index=df_train.index, dtype='float64',columns=['pca_per_1','pca_per_2','pca_per_3','pca_per_4','pca_per_5'])

df_pca_loan = pd.DataFrame(data=X_pca_loan, index=df_train.index, dtype='float64',columns=['pca_loan_1','pca_loan_2','pca_loan_3','pca_loan_4','pca_loan_5'])
# PCAの結果を結合

df_train = pd.concat([df_train,df_pca_per],axis=1)

df_train = pd.concat([df_train,df_pca_loan],axis=1)
del df_pca_per

del df_pca_loan

gc.collect()
km_per = KMeans(n_clusters=10,  # クラスタの個数を指定

            init='k-means++',           # セントロイドの初期値の決め方を決定

            max_iter=1000,         # ひとつのセントロイドを用いたときの最大イテレーション回数

            tol=1e-04,               # 収束と判定するための相対的な許容誤差

            random_state=71,     # セントロイドの初期化に用いる乱数生成器の状態

           )



# クラスラベルを予測

y_km_per = km_per.fit_predict(X_pca_per[:, 0:2])
km_loan = KMeans(n_clusters=10,  # クラスタの個数を指定

            init='k-means++',           # セントロイドの初期値の決め方を決定

            max_iter=1000,         # ひとつのセントロイドを用いたときの最大イテレーション回数

            tol=1e-04,               # 収束と判定するための相対的な許容誤差

            random_state=71,     # セントロイドの初期化に用いる乱数生成器の状態

           )



# クラスラベルを予測

y_km_loan = km_per.fit_predict(X_pca_loan[:, 0:2])
# clusteringの結果を結合

df_train = pd.concat([df_train,pd.DataFrame(y_km_per,columns=['Cluster_Per'])],axis=1)

df_train = pd.concat([df_train,pd.DataFrame(y_km_loan,columns=['Cluster_Lone'])],axis=1)

# clusteringの結果から派生（人×Loanで特定）

df_train['Cluster_Per_by_Loan'] = df_train['Cluster_Per'] * df_train['Cluster_Lone']
###emp_titleはtf-idfでエンコーディング

tdidf_emp = TfidfVectorizer(max_features=128,use_idf=True)

tdidf_title = TfidfVectorizer(max_features=128,use_idf=True)

TXT_emptitle_train = tdidf_emp.fit_transform(df_train.emp_title.copy())

TXT_title_train = tdidf_title.fit_transform(df_train.title.copy())



#TXT_emptitle_train = tdidf.fit_transform(X_train.emp_title.copy())

#TXT_emptitle_val = tdidf.transform(X_val.emp_title.copy())

#TXT_emptitle_test = tdidf.transform(df_test.emp_title.copy())



#TXT_title_train = tdidf.fit_transform(X_train.title.copy())

#TXT_title_val = tdidf.transform(X_val.title.copy())

#TXT_title_test = tdidf.transform(df_test.title.copy())
df_emptitle = pd.DataFrame(data=TXT_emptitle_train.toarray(),columns=tdidf_emp.get_feature_names())

df_title = pd.DataFrame(data=TXT_title_train.toarray(),columns=tdidf_title.get_feature_names())
# TFIDFにかけたテキストをhstack ,

df_train = pd.concat([df_train,df_emptitle],axis=1)

df_train = pd.concat([df_train,df_title],axis=1)
#ここで不要なオブジェクトを解放してメモリを確保

del df_emptitle

del df_title

del df_train_

del df_test

gc.collect()
# testデータを再抽出してIDで元の順番にソートする

X_test = df_train[df_train['Train_Test']==1].copy()

X_test = X_test.sort_values('ID', ascending=True)

X_test = X_test.set_index('ID')

X_test.drop(['loan_condition','issue_d','earliest_cr_line','emp_title','title','Train_Test'],inplace=True,axis=1)
#アンサンブル：層化抽出

blending_list = ['1回目','2回目','3回目','4回目','5回目']



#結果格納用のリストを作成

result_pred  = pd.DataFrame(columns=blending_list,index=X_test.index)



#学習データ取得

X_train = df_train[df_train['Train_Test']==0].copy()

X_train = X_train.set_index('ID')

#教師ラベルを取得

y_train = X_train['loan_condition'].copy()

#不要な列削除

X_train.drop(['loan_condition','issue_d','earliest_cr_line','emp_title','title','Train_Test'],inplace=True,axis=1)



#メモリ解放

del df_train

gc.collect



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    #print('i：',i)

    #LightGBMのインスタンスを立てる

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    #今回のバリデーションでfit

    clf.fit(X_train_, y_train_, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_val, y_val)])

        

    #予測して配列に格納

    y_pred_ = clf.predict_proba(X_test)[:,1]

    result_pred[str(i+1)+'回目'] = y_pred_



    del clf

    del X_train_

    del y_train_

    del X_val

    del y_val

    gc.collect()

'''

#アンサンブル（正式にはブレンディング）の期間を指定（時系列スプリット）

blending_list = ['Jan-2013','Jan-2014','Jan-2015']



#結果格納用のリストを作成

#result_pred = np.empty((len(X_test),3), 'float64')

result_pred  = pd.DataFrame(columns=blending_list,index=X_test.index)



for index, item in enumerate(blending_list): #リストのインデックスとアイテムを取得

    #一旦テストデータを全部取得

    tmp_train = df_train[df_train['Train_Test']==0].copy()

    tmp_train = tmp_train.set_index('ID')

    

    #時系列方向のクロスバリデーション(K=3)を行って,アンサンブルする。（次）

    #X_train = tmp_train[pd.to_datetime(tmp_train['issue_d'].astype(str)) < pd.to_datetime(item)].copy()

    X_train = tmp_train[tmp_train['issue_d'] < pd.to_datetime(item)].copy()

    if index == len(blending_list) - 1:

        X_val = tmp_train[tmp_train['issue_d'] >= pd.to_datetime(item)].copy()

    else:

        X_val = tmp_train[(tmp_train['issue_d'] >= pd.to_datetime(item)) & 

                          (tmp_train['issue_d'] < pd.to_datetime(blending_list[(index + 1)]))].copy()

    #教師ラベルを取得

    y_train = X_train['loan_condition'].copy()

    y_val = X_val['loan_condition'].copy()

    

    #不要な列を削除

    X_train.drop(['loan_condition','issue_d','earliest_cr_line','emp_title','title','Train_Test'],inplace=True,axis=1)

    X_val.drop(['loan_condition','issue_d','earliest_cr_line','emp_title','title','Train_Test'],inplace=True,axis=1)

    

    #ここで不要なオブジェクトを解放してメモリを確保

    del tmp_train

    gc.collect()

    

    #LightGBMのインスタンスを立てる

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    #今回のバリデーションでfit

    clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    

    #予測して配列に格納

    y_pred_ = clf.predict_proba(X_test)[:,1]

    result_pred[item] = y_pred_

    #tmp_result = pd.DataFrame(y_pred_)

    #result_pred = pd.concat([result_pred,tmp_result],axis=1)

'''
result_pred
#全体の平均値を予測値とする

y_pred = result_pred.mean(axis=1)
y_pred
#ハイパーパラメータチューニングを考える（まだ早いかも。）

'''

def objective(space):

    scores = []



    clf = LGBMClassifier(n_estimators=9999, **space) 



    clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

        

    scores = np.array(scores)

    print(scores.mean())

    

    return -scores.mean()

'''
'''

space ={

        'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

        'subsample': hp.uniform ('subsample', 0.8, 1),

        'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),

        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

    }

'''
'''

trials = Trials()



best = fmin(fn=objective,

              space=space, 

              algo=tpe.suggest,

              max_evals=50, 

              trials=trials, 

              rstate=np.random.RandomState(0) 

             )

'''
#LGBMClassifier(**best)
#ベストスコアで再構成

'''

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.5,

                                importance_type='split', learning_rate=0.025, max_depth=19,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=0.9996696876535358, subsample_for_bin=200000, subsample_freq=0)

'''
#clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
#X_train_sparse = X_train.tocsr()

#clf = GradientBoostingClassifier(random_state=0,max_depth=7,learning_rate=0.1,subsample=0.5,max_features=2)
#全データで再学習するため、trainとvalをくっつけて復元する

#X_train_ = pd.concat([X_train,X_val])

#y_train_ = pd.concat([y_train,y_val])
#再学習

#clf.fit(X_train_, y_train_, eval_metric='auc')

#clf.fit(X_train_sparse, y_train.values)
#全データで再学習

#y_train = df_train['loan_condition'].copy()

#X_train = df_train.drop(['loan_condition','issue_d'],axis=1)
#clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
#予測する

#y_pred = clf.predict_proba(X_test)[:,1]

# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
y_pred
#importanceを確認する

#feature = clf.feature_importances_

#imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance'])
#fig, ax = plt.subplots(figsize=(5, 8))

#lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
#特徴量減らす場合

#use_col = imp.index[:10]

#use_col
'''

#特徴量の重要度を上から順に出力する

f = pd.DataFrame({'number': range(0, len(feature)),'feature': feature[:]})

f2 = f.sort_values('feature',ascending=False)

f3 = f2.ix[:, 'number']



#特徴量の名前

label = df_test.columns[0:]



#特徴量の重要度順（降順）

indices = np.argsort(feature)[::-1]



for i in range(len(feature)):

    print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))



plt.title('Feature Importance')

plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')

plt.xticks(range(len(feature)), label[indices], rotation=90)

plt.xlim([-1, len(feature)])

plt.tight_layout()

plt.show()

'''