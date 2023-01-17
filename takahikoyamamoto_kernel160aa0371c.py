import numpy as np
import scipy as sp
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder
from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler, quantile_transform
import seaborn as sns
import datetime

import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss, roc_curve, confusion_matrix, plot_roc_curve
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from lightgbm import LGBMClassifier

# NN
# from tensorflow.keras.layers import Dense ,Dropout, BatchNormalization, Input, Embedding, SpatialDropout1D, Reshape, Concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.metrics import AUC

from hyperopt import fmin, tpe, hp, rand, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
# import xgboost as xgb
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。
#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。
# df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, skiprows=lambda x: x%20!=0)
df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0)
df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0)
# DataFrameのshapeで行数と列数を確認してみましょう。
df_train.shape, df_test.shape
#最大表示数の設定
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
# 先頭5行をみてみます。
df_train.head()
df_train.tail()
df_test.head()
df_test.tail()
# 緯度経度
df_lat = pd.read_csv('../input/homework-for-students4plus/statelatlong.csv', index_col=0)
# 結合
df_train_lat=pd.merge(df_train, df_lat, left_on='addr_state', right_on='State', how='left')
df_test_lat=pd.merge(df_test, df_lat, left_on='addr_state', right_on='State', how='left')
df_train=df_train_lat
df_test=df_test_lat
# GDP
df_gdp = pd.read_csv('../input/homework-for-students4plus/US_GDP_by_State.csv', index_col=0)
# リーケージ回避のため、借りた日の前年のGDPと結合にする。申請の年に-1する
df_train['issue_d'].str[-4:].astype(int)-1
df_test['issue_d'].str[-4:].astype(int)-1
df_train=df_train.assign(issue_y_pre=df_train['issue_d'].str[-4:].astype(int)-1)
df_test=df_test.assign(issue_y_pre=df_test['issue_d'].str[-4:].astype(int)-1)
# 結合
df_train_gdp=pd.merge(df_train, df_gdp, left_on=['City','issue_y_pre'], right_on=['State','year'], how='left')
df_test_gdp=pd.merge(df_test, df_gdp, left_on=['City','issue_y_pre'], right_on=['State','year'], how='left')
df_train=df_train_gdp
df_test=df_test_gdp
# S&P Index
df_spi = pd.read_csv('../input/homework-for-students4plus/spi.csv', index_col=0, parse_dates=[0])
# issue_dをyyyy-mm形式に変更
df_train['issue_d']=pd.to_datetime(df_train['issue_d']).dt.strftime('%Y-%m')
df_test['issue_d']=pd.to_datetime(df_test['issue_d']).dt.strftime('%Y-%m')
# spiを月別集計
df_spi_m=df_spi.resample(rule="M").mean()
# index化したdate列を列に戻す
df_spi_m=df_spi_m.reset_index()
# yyyy-mm形式に変更
df_spi_m['date']=df_spi_m['date'].dt.strftime('%Y-%m')
# 結合
df_train_spi=pd.merge(df_train, df_spi_m, left_on='issue_d', right_on='date', how='left')
df_test_spi=pd.merge(df_test, df_spi_m, left_on='issue_d', right_on='date', how='left')
df_train=df_train_spi
df_test=df_test_spi
df_train.shape,df_test.shape
df_train['issue_d'] = pd.to_datetime(df_train['issue_d'])
df_test['issue_d'] = pd.to_datetime(df_test['issue_d'])
# issue_dを年に分解
df_train['issue_d_year'] = df_train['issue_d'].dt.year
df_test['issue_d_year'] = df_test['issue_d'].dt.year
# issue_dを月に分解
df_train['issue_d_month'] = df_train['issue_d'].dt.month
df_test['issue_d_month'] = df_test['issue_d'].dt.month
# Feature Engineering
df_train['loan_month']=df_train['loan_amnt']/df_train['installment'] #返済するまでの月
df_train['loan_end_date']=df_train['issue_d']+pd.to_timedelta(df_train['loan_month']*30,unit='D') #返済後の日付
df_train['inst_late']=df_train['installment']*12/(df_train['annual_inc']+1) #年収に対する返済額の割合
df_train['line_late']=df_train['open_acc']/df_train['total_acc'] #クレジットライン総数に対するオープンの割合
df_train['earliest_cr_line_y']=pd.to_datetime(df_train['earliest_cr_line'],format='%b-%Y').dt.year
df_train['earliest_cr_line_m']=pd.to_datetime(df_train['earliest_cr_line'],format='%b-%Y').dt.month

df_test['loan_month']=df_test['loan_amnt']/df_test['installment'] #返済するまでの月
df_test['loan_end_date']=df_test['issue_d']+pd.to_timedelta(df_test['loan_month']*30,unit='D') #返済後の日付
df_test['inst_late']=df_test['installment']*12/(df_test['annual_inc']+1) #年収に対する返済額の割合 分母0を防ぐため+1
df_test['line_late']=df_test['open_acc']/df_test['total_acc'] #クレジットライン総数に対するオープンの割合
df_test['earliest_cr_line_y']=pd.to_datetime(df_test['earliest_cr_line'],format='%b-%Y').dt.year
df_test['earliest_cr_line_m']=pd.to_datetime(df_test['earliest_cr_line'],format='%b-%Y').dt.month
# emp_length　数値型に変換
emp_length_dic = {'4 years':4, '1 year':1, '10+ years':15, '3 years':3, '2 years':2, '6 years':6,
       '8 years':8, '5 years':5, '9 years':9, '< 1 year':0.5, '7 years':7}
df_train['emp_length'] = df_train['emp_length'].map(emp_length_dic).fillna(0)
df_test['emp_length'] = df_test['emp_length'].map(emp_length_dic).fillna(0)
# grade
grade_dic = {
        "A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6
}
df_train['grade'] = df_train['grade'].map(grade_dic)
df_test['grade'] = df_test['grade'].map(grade_dic)
# sub_grade
sub_grade_dic = {
    "A1": 0,"A2": 1,"A3": 2,"A4": 3,"A5": 4,
    "B1": 5,"B2": 6,"B3": 7,"B4": 8,"B5": 9,
    "C1": 10,"C2": 11,"C3": 12,"C4": 13,"C5": 14,
    "D1": 15,"D2": 16,"D3": 17,"D4": 18,"D5": 19,
    "E1": 20,"E2": 21,"E3": 22,"E4": 23,"E5": 24,
    "F1": 25,"F2": 26,"F3": 27,"F4": 28,"F5": 29,
    "G1": 30,"G2": 31,"G3": 32,"G4": 33,"G5": 34
}
df_train['sub_grade'] = df_train['sub_grade'].map(sub_grade_dic)
df_test['sub_grade'] = df_test['sub_grade'].map(sub_grade_dic)
ep=0.01 #分母が0にならないように
df_train['annual_inc_acc_now_delinq']= df_train.annual_inc/(df_train.acc_now_delinq+ep)
df_train['annual_inc_open_acc']= df_train.annual_inc/(df_train.open_acc+ep)
df_train['annual_inc_total_acc']= df_train.annual_inc/(df_train.total_acc+ep)
df_train['annual_inc_loan_amnt']= df_train.annual_inc/(df_train.loan_amnt+ep)
df_train['annual_inc_emp_length']= df_train.annual_inc/(df_train.emp_length+ep)
df_train['installment_tot_cur_bal']= df_train.installment/(df_train.tot_cur_bal+ep)
df_train['installment_acc_now_delinq']= df_train.installment/(df_train.acc_now_delinq+ep)
df_train['installment_open_acc']= df_train.installment/(df_train.open_acc+ep)
df_train['installment_total_acc']= df_train.installment/(df_train.total_acc+ep)
df_train['installment_emp_length']= df_train.installment/(df_train.emp_length+ep)
df_train['tot_cur_bal_acc_now_delinq']= df_train.tot_cur_bal/(df_train.acc_now_delinq+ep)
df_train['tot_cur_bal_total_acc']= df_train.tot_cur_bal/(df_train.total_acc+ep)
df_train['tot_cur_bal_loan_amnt']= df_train.tot_cur_bal/(df_train.loan_amnt+ep)
df_train['tot_cur_bal_emp_length']= df_train.tot_cur_bal/(df_train.emp_length+ep)
df_train['acc_now_delinq_open_acc']= df_train.acc_now_delinq/(df_train.open_acc+ep)
df_train['acc_now_delinq_total_acc']= df_train.acc_now_delinq/(df_train.total_acc+ep)
df_train['acc_now_delinq_loan_amnt']= df_train.acc_now_delinq/(df_train.loan_amnt+ep)
df_train['acc_now_delinq_emp_length']= df_train.acc_now_delinq/(df_train.emp_length+ep)
df_train['open_acc_loan_amnt']= df_train.open_acc/(df_train.loan_amnt+ep)
df_train['open_acc_emp_length']= df_train.open_acc/(df_train.emp_length+ep)
df_train['open_acc_loan_amnt']= df_train.total_acc/(df_train.loan_amnt+ep)
df_train['open_acc_emp_length']= df_train.total_acc/(df_train.emp_length+ep)
df_test['annual_inc_acc_now_delinq']= df_test.annual_inc/(df_test.acc_now_delinq+ep)
df_test['annual_inc_open_acc']= df_test.annual_inc/(df_test.open_acc+ep)
df_test['annual_inc_total_acc']= df_test.annual_inc/(df_test.total_acc+ep)
df_test['annual_inc_loan_amnt']= df_test.annual_inc/(df_test.loan_amnt+ep)
df_test['annual_inc_emp_length']= df_test.annual_inc/(df_test.emp_length+ep)
df_test['installment_tot_cur_bal']= df_test.installment/(df_test.tot_cur_bal+ep)
df_test['installment_acc_now_delinq']= df_test.installment/(df_test.acc_now_delinq+ep)
df_test['installment_open_acc']= df_test.installment/(df_test.open_acc+ep)
df_test['installment_total_acc']= df_test.installment/(df_test.total_acc+ep)
df_test['installment_emp_length']= df_test.installment/(df_test.emp_length+ep)
df_test['tot_cur_bal_acc_now_delinq']= df_test.tot_cur_bal/(df_test.acc_now_delinq+ep)
df_test['tot_cur_bal_total_acc']= df_test.tot_cur_bal/(df_test.total_acc+ep)
df_test['tot_cur_bal_loan_amnt']= df_test.tot_cur_bal/(df_test.loan_amnt+ep)
df_test['tot_cur_bal_emp_length']= df_test.tot_cur_bal/(df_test.emp_length+ep)
df_test['acc_now_delinq_open_acc']= df_test.acc_now_delinq/(df_test.open_acc+ep)
df_test['acc_now_delinq_total_acc']= df_test.acc_now_delinq/(df_test.total_acc+ep)
df_test['acc_now_delinq_loan_amnt']= df_test.acc_now_delinq/(df_test.loan_amnt+ep)
df_test['acc_now_delinq_emp_length']= df_test.acc_now_delinq/(df_test.emp_length+ep)
df_test['open_acc_loan_amnt']= df_test.open_acc/(df_test.loan_amnt+ep)
df_test['open_acc_emp_length']= df_test.open_acc/(df_test.emp_length+ep)
df_test['open_acc_loan_amnt']= df_test.total_acc/(df_test.loan_amnt+ep)
df_test['open_acc_emp_length']= df_test.total_acc/(df_test.emp_length+ep)
df_train.head()
df_test
df_train.describe()
df_test.describe()
df_train.dtypes
df_test.dtypes
df_train.shape,df_test.shape
pd.crosstab(pd.to_datetime(df_train['issue_d'], format='%Y-%m-%d').dt.strftime('%Y'), df_train['loan_condition'])
# 2013年末以前のデータは削除
df_train=df_train[df_train['issue_d'] > datetime.datetime(2013,12,31)]
pd.crosstab(pd.to_datetime(df_train['issue_d'], format='%Y-%m-%d').dt.strftime('%Y'), df_train['loan_condition'])
df_train['earliest_cr_line'] = pd.to_datetime(df_train['earliest_cr_line'],format='%b-%Y')
df_test['earliest_cr_line'] = pd.to_datetime(df_test['earliest_cr_line'],format='%b-%Y')
# 日付型を数値に変換
df_train['issue_d']=df_train['issue_d'].astype(np.int64)
df_train['loan_end_date']=df_train['loan_end_date'].astype(np.int64)
df_train['earliest_cr_line']=df_train['earliest_cr_line'].astype(np.int64)

df_test['issue_d']=df_test['issue_d'].astype(np.int64)
df_test['loan_end_date']=df_test['loan_end_date'].astype(np.int64)
df_test['earliest_cr_line']=df_test['earliest_cr_line'].astype(np.int64)
df_train.head()
df_test.head()
df_train.describe()
df_train.shape,df_test.shape
df_train[df_train.loan_condition==1].loan_amnt.mean() # 貸し倒れたローンの平均額
# 上の貸し倒れたローンに対するものを参考に、貸し倒れていないローンの平均額を算出みてください。
df_train.loan_condition==0
df_train[df_train.loan_condition==0].loan_amnt.mean() # 貸し倒れていないローンの平均額
df_train[df_train.emp_title.isnull()==False].loan_amnt.mean() #emp_titleがnullでない人のローンの平均額
f = 'loan_amnt'

plt.figure(figsize=[7,7])
df_train[f].hist(density=True, alpha=0.5, bins=50,color='r') # α:透過率 colorがなくてもいい
df_test[f].hist(density=True, alpha=0.5, bins=50,color='b') # colorがなくてもいい
# testデータに対する可視化を記入してみましょう
plt.xlabel(f)
plt.ylabel('density') #　density:絶対値でなく相対値
plt.show()
df_train[f].value_counts() #何か買いたい商品に合わせて借りている可能性
f = 'purpose'
# value_countsを用いてtrainのpurposeに対して集計結果をみてみましょう。
df_train[f].value_counts()
df_train[f].value_counts()/len(df_train) #相対で見る
# 同様にtestデータに対して(df_test)
df_test[f].value_counts()
df_test[f].value_counts()/len(df_test) #相対で見る
df_train.shape,df_test.shape
# 可視化
# df_train_corr=df_train.corr().style.background_gradient(axis=None)
# df_train_corr.to_excel('df_train_corr.xlsx')
# 他の特徴量と相関係数|0.95|以上の特徴量は削除
df_train.drop(['installment'], axis=1, inplace=True)
df_train.drop(['grade'], axis=1, inplace=True)
df_train.drop(['annual_inc'], axis=1, inplace=True)
df_train.drop(['earliest_cr_line'], axis=1, inplace=True)
df_train.drop(['tot_cur_bal'], axis=1, inplace=True)
df_train.drop(['issue_y_pre'], axis=1, inplace=True)
df_train.drop(['State & Local Spending'], axis=1, inplace=True)
df_train.drop(['Gross State Product'], axis=1, inplace=True)
df_train.drop(['year'], axis=1, inplace=True)
# 他の特徴量と相関係数|0.95|以上の特徴量は削除
df_test.drop(['installment'], axis=1, inplace=True)
df_test.drop(['grade'], axis=1, inplace=True)
df_test.drop(['annual_inc'], axis=1, inplace=True)
df_test.drop(['earliest_cr_line'], axis=1, inplace=True)
df_test.drop(['tot_cur_bal'], axis=1, inplace=True)
df_test.drop(['issue_y_pre'], axis=1, inplace=True)
df_test.drop(['State & Local Spending'], axis=1, inplace=True)
df_test.drop(['Gross State Product'], axis=1, inplace=True)
df_test.drop(['year'], axis=1, inplace=True)
y_train = df_train.loan_condition #目的変数
X_train = df_train.drop(['loan_condition'], axis=1) #説明変数

X_test = df_test
X_train
X_test
f='loan_amnt'
scaler = StandardScaler()
scaler.fit(X_train[[f]])
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。
cats = []
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        cats.append(col)
        
        print(col, X_train[col].nunique())
# dtypeが数値のカラム名とユニーク数を確認してみましょう。
num_cols = []
for col in X_train.columns:
    if X_train[col].dtype != 'object':
        num_cols.append(col)
        
        print(col, X_train[col].nunique())
# 欠損値フラグ作ってみる
nullflagcol = X_train.columns[X_train.isnull().sum()!=0].values 
nullflagcol
for col in nullflagcol:
    # 元の列(col)のうち欠損している行を抽出。その行に対しcolnull列=1を入れる    
    X_train.loc[X_train[col].isnull(),col+'null']=1
    # 0埋め    
    X_train[col+'null']=X_train[col+'null'].fillna(0)
    # 元の列(col)のうち欠損している行を抽出。その行に対しcolnull列=1を入れる    
    X_test.loc[X_test[col].isnull(),col+'null']=1
    # 0埋め    
    X_test[col+'null']=X_test[col+'null'].fillna(0)    
# X_train[X_train['mths_since_last_delinqnull']==0].head()
# X_train[X_train['mths_since_last_delinqnull']==1].head()
X_train.shape,X_test.shape
X_train.describe()
X_test.describe()
# 2σを基準に外れ値を埋める関数
def outlier_2s(df):

    for i in range(len(df.columns)):
        # 列を抽出する
        col = df.iloc[:,i]
        # 平均と標準偏差
        average = np.mean(col)
        sd = np.std(col)
        # 外れ値の基準点
        outlier_min = average - (sd) * 2
        outlier_max = average + (sd) * 2
#         # 範囲から外れている値を除く
#         col[col < outlier_min] = None
#         col[col > outlier_max] = None
        # 範囲から外れている値をminより小さい値,maxより大きい値で埋める
        col[col < outlier_min] = np.min(col)-1
        col[col > outlier_max] = np.max(col)+1

    return df
X_train_numcols=X_train[num_cols]
X_train_numcol =outlier_2s(X_train_numcols)
X_test_numcols=X_test[num_cols]
X_test_numcol =outlier_2s(X_test_numcols)
X_train[num_cols]=X_train_numcols
X_test[num_cols]=X_test_numcols
X_train.describe()
X_test.describe()
# 対数変換
# X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)
# X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)
# ローン額をRankGaussしてみる
# 学習データとテストデータを結合した上でRankGaussによる変換を実施
X_all = pd.concat([X_train, X_test], axis=0)

X_all['loan_amnt'] = quantile_transform(X_all['loan_amnt'].to_frame(),
n_quantiles=100, random_state=0, output_distribution='normal')

# 学習データとテストデータに再分割
X_train = X_all.iloc[:X_train.shape[0], :]
X_test = X_all.iloc[X_train.shape[0]:, :]
# mths_since_last_recordをRankGaussしてみる

# 学習データとテストデータを結合した上でRankGaussによる変換を実施
X_all = pd.concat([X_train, X_test], axis=0)

X_all['mths_since_last_record'] = quantile_transform(X_all['mths_since_last_record'].to_frame(),
n_quantiles=100, random_state=0, output_distribution='normal')

# 学習データとテストデータに再分割
X_train = X_all.iloc[:X_train.shape[0], :]
X_test = X_all.iloc[X_train.shape[0]:, :]
# # 学習データとテストデータを結合した上でRankGaussによる変換を実施
# X_all = pd.concat([X_train, X_test], axis=0)

# X_all[num_cols] = quantile_transform(X_all[num_cols],
# n_quantiles=100, random_state=0, output_distribution='normal')

# # 学習データとテストデータに再分割
# X_train = X_all.iloc[:X_train.shape[0], :]
# X_test = X_all.iloc[X_train.shape[0]:, :]
# X_train
# X_test
# X_train.describe()
# X_test.describe()
X_train['emp_title'].head(10) # カテゴリよりテキストとして扱ったほうが良いかもしれない
X_train['emp_title'].value_counts()
X_train['emp_title'].value_counts().hist(bins=100)
# col = 'purpose'

# encoder = OneHotEncoder()
# enc_train = encoder.fit_transform(X_train[col].values) #fitとtransformを分ける書き方もある
# enc_test = encoder.transform(X_test[col].values)
# enc_train.head()
# enc_test.head()
# Onehotの例を参考にやってみましょう
# https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
# # col = 'purpose'

# col ='grade'

# encoder = OrdinalEncoder()
# enc_train = encoder.fit_transform(X_train[col].values) #fitとtransformを分ける書き方もある
# enc_test = encoder.transform(X_test[col].values)
# enc_train.head(30)
# enc_test.head(30)
# pd.crosstab(df_train['grade'], df_train['loan_condition'])
# pd.crosstab(df_train['sub_grade'], df_train['loan_condition'])
# # value_couontsで集計した結果を、
# summary = X_train[col].value_counts()
# summary
# # mapする。
# enc_train = X_train[col].map(summary)
# enc_test = X_test[col].map(summary)
# enc_train.head()
# enc_test.head()
target = 'loan_condition'
X_temp = pd.concat([X_train, y_train], axis=1)

# X_testはX_trainでエンコーディングする
summary = X_temp.groupby([col])[target].mean()
enc_test = X_test[col].map(summary) 

    
# X_trainのカテゴリ変数をoofでエンコーディングする
# ここでは理解のために自分で交差検定的に実施するが、xfeatなどを用いても良い
skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):
    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]
    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

    summary = X_train_.groupby([col])[target].mean()
    enc_train.iloc[val_ix] = X_val[col].map(summary)
enc_train
enc_test
TXT_train = X_train.emp_title.copy()
TXT_test = X_test.emp_title.copy()
cats.remove('emp_title')
TXT_train
# oe = OrdinalEncoder(cols=cats, return_df=False)

# X_train[cats] = oe.fit_transform(X_train[cats])
# X_test[cats] = oe.transform(X_test[cats])
# 以下を参考に自分で書いてみましょう 
X_train.drop(['emp_title'], axis=1, inplace=True)
X_test.drop(['emp_title'], axis=1, inplace=True)

X_train['mths_since_last_delinq'].fillna(-9999, inplace=True)
X_test['mths_since_last_delinq'].fillna(-9999, inplace=True)
X_train['mths_since_last_record'].fillna(-9999, inplace=True)
X_test['mths_since_last_record'].fillna(-9999, inplace=True)
X_train['mths_since_last_major_derog'].fillna(-9999, inplace=True)
X_test['mths_since_last_major_derog'].fillna(-9999, inplace=True)
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
# Target Encoding
target = 'loan_condition'
X_temp = pd.concat([X_train, y_train], axis=1)

for col in cats:

    # X_testはX_trainでエンコーディングする
    summary = X_temp.groupby([col])[target].mean()
    X_test[col] = X_test[col].map(summary) 


    # X_trainのカテゴリ変数をoofでエンコーディングする
    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):
        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]
        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = X_train_.groupby([col])[target].mean()
        enc_train.iloc[val_ix] = X_val[col].map(summary)
        
    X_train[col]  = enc_train
X_train.fillna(X_train.mean(), axis=0, inplace=True)
X_test.fillna(X_train.mean(), axis=0, inplace=True)
# 学習用と検証用に分割する
# X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=43)
# clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7000000000000001,
#                             importance_type='split', learning_rate=0.05, max_depth=10,
#                             min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
#                             n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
#                             random_state=0, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#                             subsample=0.9929417385040324, subsample_for_bin=200000, subsample_freq=0)

# clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])
# imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)
# imp
# fig, ax = plt.subplots(figsize=(5, 16))
# lgb.plot_importance(clf, ax=ax, importance_type='gain')
# # importance 1000以下を削除
# X_train.drop(['close'], axis=1, inplace=True)
# X_train.drop(['initial_list_status'], axis=1, inplace=True)
# X_train.drop(['issue_d_month'], axis=1, inplace=True)
# X_train.drop(['pub_rec'], axis=1, inplace=True)
# X_train.drop(['application_type'], axis=1, inplace=True)
# X_train.drop(['mths_since_last_major_derognull'], axis=1, inplace=True)
# X_train.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)
# X_train.drop(['mths_since_last_recordnull'], axis=1, inplace=True)
# X_train.drop(['acc_now_delinq_emp_length'], axis=1, inplace=True)
# X_train.drop(['issue_d_year'], axis=1, inplace=True)
# X_train.drop(['mths_since_last_delinqnull'], axis=1, inplace=True)
# # importance 1000以下を削除
# X_test.drop(['close'], axis=1, inplace=True)
# X_test.drop(['initial_list_status'], axis=1, inplace=True)
# X_test.drop(['issue_d_month'], axis=1, inplace=True)
# X_test.drop(['pub_rec'], axis=1, inplace=True)
# X_test.drop(['application_type'], axis=1, inplace=True)
# X_test.drop(['mths_since_last_major_derognull'], axis=1, inplace=True)
# X_test.drop(['collections_12_mths_ex_med'], axis=1, inplace=True)
# X_test.drop(['mths_since_last_recordnull'], axis=1, inplace=True)
# X_test.drop(['acc_now_delinq_emp_length'], axis=1, inplace=True)
# X_test.drop(['issue_d_year'], axis=1, inplace=True)
# X_test.drop(['mths_since_last_delinqnull'], axis=1, inplace=True)
TXT_train
# 大文字小文字で別扱いかもしれない
TXT_train = TXT_train.str.upper()
TXT_test = TXT_test.str.upper()
TXT_train.fillna('#', inplace=True)
TXT_test.fillna('#', inplace=True)
TXT_train
tfidf = TfidfVectorizer(max_features=100, use_idf=True)
TXT_train_enc = tfidf.fit_transform(TXT_train)
TXT_test_enc = tfidf.transform(TXT_test)
#疎行列が帰ってきます。
TXT_train_enc
# todenseで密行列に変換できますが、ほどんどゼロであることがみて取れます。
TXT_train_enc.todense()
X_train_hstack=sp.sparse.hstack([X_train.values, TXT_train_enc])
X_test_hstack=sp.sparse.hstack([X_test.values, TXT_test_enc])
X_train_text=pd.DataFrame.sparse.from_spmatrix(X_train_hstack)
X_test_text=pd.DataFrame.sparse.from_spmatrix(X_test_hstack)
X_train_text.head()
X_test_text.head()
X_train=X_train_text
X_test=X_test_text
# df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)
# #df_train = pd.read_csv('../input/train.csv', index_col=0)
# y_train = df_train.loan_condition
# X_train = df_train.drop(['loan_condition'], axis=1)

# X_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0, parse_dates=['issue_d'])
# X_train
# y_train
# X_test
# cat = []
# num = []

# for col in X_train.columns:
#     if X_train[col].dtype == 'object':
#         if col != 'emp_title':
#             cat.append(col)
#     else:
#         if col != 'issue_d':
#             num.append(col)
# # train/test
# # 特徴量タイプごとに分割する
# cat_train = X_train[cat]
# txt_train = X_train.emp_title
# num_train = X_train[num]

# cat_test = X_test[cat]
# txt_test = X_test.emp_title
# num_test = X_test[num]
# scaler = StandardScaler()
# num_train = scaler.fit_transform(num_train.fillna(num_train.median()))
# num_test = scaler.transform(num_test.fillna(num_test.median()))
# for col in tqdm(cat):
#     oe = OrdinalEncoder(return_df=False)
    
#     cat_train[col] = oe.fit_transform(cat_train[[col]])
#     cat_test[col] = oe.transform(cat_test[[col]]) 
# len(num_train),len(num_test)
# # バラしてリストにする
# cat_train = [cat_train.values[:, k].reshape(-1,1) for k in range(len(cat))]
# cat_test = [cat_test.values[:, k].reshape(-1,1) for k in range(len(cat))]
# cat_train
# # numericとcategoricalを結合して一つのリストにする
# X_train = [num_train] + cat_train
# X_test = [num_test] + cat_test
# X_train
# X_test
# y_train
# # シンプルなNN
# def create_model():
#     num= Input(shape=(num_train.shape[1],))
#     out_num = Dense(194, activation='relu')(num)
#     out_num = BatchNormalization()(out_num)
#     out_num = Dropout(0.5)(out_num)
    
#     inputs = [num]
#     outputs = [out_num]
    
#     # カテゴリカル変数はembeddingする
#     for c in cat:
#         num_unique_values = int(df_train[c].nunique())
#         emb_dim = int(min(np.ceil((num_unique_values)/2), 50))
#         inp = Input(shape=(1,))
#         inputs.append(inp)
#         out = Embedding(num_unique_values+2, emb_dim, name=c)(inp)
#         out = SpatialDropout1D(0.3)(out)
#         out = Reshape((emb_dim, ))(out)
#         outputs.append(out)
    
#     x = Concatenate()(outputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     outp = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=inputs, outputs=outp)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])
    
#     return model
# X_test.shape[0]
# model = create_model()

# es = EarlyStopping(monitor='val_loss', patience=0)

# model.fit(X_train, y_train, batch_size=32, epochs=999, validation_split=0.2, callbacks=[es])
# model.predict(X_test).ravel()
# def objective(space):
#     scores = []

#     skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#     for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
#         X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
#         X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

#         clf = LGBMClassifier(n_estimators=9999, **space) 

#         clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
#         y_pred = clf.predict_proba(X_val)[:,1]
#         score = roc_auc_score(y_val, y_pred)
#         scores.append(score)
        
#     scores = np.array(scores)
#     print(scores.mean())
    
#     return -scores.mean()
# space ={
#         'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),
#         'subsample': hp.uniform ('subsample', 0.8, 1),
#         'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),
#         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)
#     }
# trials = Trials()

# best = fmin(fn=objective,
#               space=space, 
#               algo=tpe.suggest,
#               max_evals=20, 
#               trials=trials, 
#               rstate=np.random.RandomState(71) 
#              )
# LGBMClassifier(**best)
# trials.best_trial['result']
# # 今度はカテゴリ特徴量も追加してモデリングしてみましょう。
# # CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。
# # 数値 中央値,-9999埋め
# # カテゴリ TargetEncoding
# # テキスト 'emp_title' 100字
# LightGB CV Averaging random1
scores = []
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=10, random_state=30, shuffle=True)

# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
#     X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
#     X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]
for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):
    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7000000000000001,
                                importance_type='split', learning_rate=0.05, max_depth=10,
                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
                                random_state=0, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                subsample=0.9929417385040324, subsample_for_bin=200000, subsample_freq=0)
    
    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])
    y_pred = clf.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(y_val, y_pred))
    
    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく

scores = np.array(scores)
print('Ave. CV score is %f' % scores.mean())
score1 = scores.mean()

y_pred_test /= 10 # 最後にfold数で割る
y_pred1 = y_pred_test
# LightGB CV Averaging random2
scores = []
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=10, random_state=60, shuffle=True)

# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
#     X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
#     X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]
for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):
    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7000000000000001,
                                importance_type='split', learning_rate=0.05, max_depth=10,
                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
                                random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                subsample=0.9929417385040324, subsample_for_bin=200000, subsample_freq=0)
    
    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])
    y_pred = clf.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(y_val, y_pred))
    
    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく

scores = np.array(scores)
print('Ave. CV score is %f' % scores.mean())
score2 = scores.mean()

y_pred_test /= 10 # 最後にfold数で割る
y_pred2 = y_pred_test
# LightGB CV Averaging random3
scores = []
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=10, random_state=85, shuffle=True)

# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
#     X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
#     X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]
for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):
    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7000000000000001,
                                importance_type='split', learning_rate=0.05, max_depth=10,
                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,
                                random_state=100, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                subsample=0.9929417385040324, subsample_for_bin=200000, subsample_freq=0)
    
    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])
    y_pred = clf.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(y_val, y_pred))
    
    y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく

scores = np.array(scores)
print('Ave. CV score is %f' % scores.mean())
score3 = scores.mean()

y_pred_test /= 10 # 最後にfold数で割る
y_pred3 = y_pred_test
# 平均スコアを算出 
# np.array(scores).mean()
(score1+score2+score3)/3
# アンサンブル
y_pred = (y_pred1+y_pred2+y_pred3)/3
# sample submissionを読み込んで、予測値を代入の後、保存する
submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred
# submission.loan_condition = y_pred_test
submission.to_csv('submission.csv')
submission.head() # まずは初回submitしてみましょう！これからこのモデルの改善を進めていくことになります。
### The end of this notebook ###