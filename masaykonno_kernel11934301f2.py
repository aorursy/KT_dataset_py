import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy as sp
import pandas as pd
from pandas import DataFrame, Series

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm import LGBMClassifier
pd.set_option('display.max_columns', 50)
#読み込み

#df_train = pd.read_csv('/kaggle/input/homework-for-students4plus/train.csv', index_col=0, skiprows=lambda x: x%10!=0)
df_train = pd.read_csv('/kaggle/input/homework-for-students4plus/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/homework-for-students4plus/test.csv', index_col=0)
#⭐️
#issue_d を数値に変換する処理

#df_train['earliest_cr_line'].str[-4:]
df_train_earliest = pd.merge(df_train['issue_d'].str[-4:], df_train,left_index=True, right_index=True)
df_train_earliest = df_train_earliest.rename(columns = {'issue_d_x':'issue_d_year'})
df_train_earliest = pd.merge(df_train['issue_d'].str[:3], df_train_earliest,left_index=True, right_index=True)
df_train_earliest = df_train_earliest.rename(columns = {'issue_d':'issue_d_month'})
df_train_earliest = df_train_earliest.rename(columns = {'issue_d_y':'issue_d'})

#df_test['earliest_cr_line'].str[-4:]
df_test_earliest = pd.merge(df_test['issue_d'].str[-4:], df_test,left_index=True, right_index=True)
df_test_earliest = df_test_earliest.rename(columns = {'issue_d_x':'issue_d_year'})
df_test_earliest = pd.merge(df_test['issue_d'].str[:3], df_test_earliest,left_index=True, right_index=True)
df_test_earliest = df_test_earliest.rename(columns = {'issue_d':'issue_d_month'})
df_test_earliest = df_test_earliest.rename(columns = {'issue_d_y':'issue_d'})

df_train = df_train_earliest
df_test =  df_test_earliest

###################
df_train['issue_d_month_mm'] = df_train['issue_d_month'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'})
df_test['issue_d_month_mm'] = df_test['issue_d_month'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'})

df_train['issue_d_yyyymm'] = df_train['issue_d_year'].str.cat(df_train['issue_d_month_mm'])
df_test['issue_d_yyyymm'] = df_test['issue_d_year'].str.cat(df_test['issue_d_month_mm'])

df_train['issue_d_yyyymm'] = df_train['issue_d_yyyymm'].fillna(-9999)
df_train['issue_d_yyyymm'] = df_train['issue_d_yyyymm'].astype(int)

df_test['issue_d_yyyymm'] = df_test['issue_d_yyyymm'].fillna(-9999)
df_test['issue_d_yyyymm'] = df_test['issue_d_yyyymm'].astype(int)

###################


#nanを０埋め、intに型変換
df_train['issue_d_year'] = df_train['issue_d_year'].fillna(-9999)
df_train['issue_d_year'] = df_train['issue_d_year'].astype(int)

df_test['issue_d_year'] = df_test['issue_d_year'].fillna(-9999)
df_test['issue_d_year'] = df_test['issue_d_year'].astype(int)

df_train['issue_d_month'] = df_train['issue_d_month'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12})
df_test['issue_d_month'] = df_test['issue_d_month'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12})

#nanを０埋め、intに型変換
df_train['issue_d_month'] = df_train['issue_d_month'].fillna(-9999)
df_train['issue_d_month'] = df_train['issue_d_month'].astype(int)

df_test['issue_d_month'] = df_test['issue_d_month'].fillna(-9999)
df_test['issue_d_month'] = df_test['issue_d_month'].astype(int)


#⭐️期間を絞る
#df_train = df_train[df_train['issue_d_year'] >= 2014]
df_train = df_train[df_train['issue_d_yyyymm'] >= 201401]

cat = []
num = []


for col in df_train.columns:

    if df_train[col].dtype == 'object':
        cat.append(col)
    else:
        num.append(col)
#新たに追加していく数値系特徴量

df_train['mths_since_last_delinq']+ df_train['mths_since_last_major_derog'] +df_train['mths_since_last_record']
#⭐️
#総借金額がnanの場合、loan_amntを代わりに使う。
df_train.loc[df_train['tot_cur_bal'].isnull(), 'tot_cur_bal'] = df_train[df_train['tot_cur_bal'].isnull()]['loan_amnt'] 
df_test.loc[df_test['tot_cur_bal'].isnull(), 'tot_cur_bal'] = df_test[df_test['tot_cur_bal'].isnull()]['loan_amnt'] 
#⭐️
#installment/annual_inc
#月の給料に対する支払い割合
#ローンが発生した場合、借り手が支払う月々の支払い/(登録時に借り手が提供する自己申告の年収。/12)

df_train['installment/annual_inc'] = df_train['installment']/(df_train['annual_inc']/12)
df_test['installment/annual_inc'] = df_test['installment']/(df_test['annual_inc']/12)

#⭐️

#loan_amnt/installment
#完済支払い回数
#借金額/月の支払い額

df_train['loan_amnt/installment'] = df_train['loan_amnt']/df_train['installment']
df_test['loan_amnt/installment'] = df_test['loan_amnt']/df_test['installment']

#⭐️特徴列追加(文字列の列同士の結合)

#addr_state

#series作成
sr_train_addr = df_train['addr_state'].str.cat(df_train['zip_code'])
sr_test_addr = df_test['addr_state'].str.cat(df_test['zip_code'])

#seriesをDF変換
df_train_addr = pd.DataFrame(sr_train_addr)
df_test_addr = pd.DataFrame(sr_test_addr)

#列名変換
df_train_addr = df_train_addr.rename(columns = {'addr_state':'address'})
df_test_addr = df_test_addr.rename(columns = {'addr_state':'address'})

#結合
df_train = pd.merge(df_train_addr, df_train,left_index=True, right_index=True)
df_test = pd.merge(df_test_addr, df_test,left_index=True, right_index=True)
#小文字化
df_train['emp_title'] = df_train['emp_title'].str.lower()
df_test['emp_title'] = df_test['emp_title'].str.lower()
#########################
#emp_title 単語数カウント

#文字列分割
df_train_emp_title_split = df_train['emp_title'].str.split(' ', expand=True)
df_test_emp_title_split = df_test['emp_title'].str.split(' ', expand=True)

#空白をNone変換
df_train_emp_title_split = df_train_emp_title_split.replace('',None) 
df_test_emp_title_split = df_test_emp_title_split.replace('',None) 

#null以外をカウント
df_train_emp_title_split_count = df_train_emp_title_split.count(axis=1)
df_test_emp_title_split_count = df_test_emp_title_split.count(axis=1)

#df変換
df_train_emp_title_split_count = pd.DataFrame(df_train_emp_title_split_count)
df_test_emp_title_split_count = pd.DataFrame(df_test_emp_title_split_count)

#列名変換
df_train_e = df_train_emp_title_split_count.rename(columns = {0:'emp_title_word_count'})
df_test_e = df_test_emp_title_split_count.rename(columns = {0:'emp_title_word_count'})

#結合
df_train = pd.merge(df_train_e, df_train,left_index=True, right_index=True)
df_test = pd.merge(df_test_e, df_test,left_index=True, right_index=True)

#########################

#欠損値の数を行ごとにカウント
df_train_nan_s = df_train.isnull().sum(axis=1)
df_test_nan_s = df_test.isnull().sum(axis=1)
#seriesをDF変換
df_train_nan = pd.DataFrame(df_train_nan_s)
df_test_nan = pd.DataFrame(df_test_nan_s)
df_train_nan = df_train_nan.rename(columns = {0:'nan_count'})
df_test_nan = df_test_nan.rename(columns = {0:'nan_count'})
df_train = pd.merge(df_train_nan, df_train,left_index=True, right_index=True)
df_test = pd.merge(df_test_nan, df_test,left_index=True, right_index=True)
#⭐️文字列系のカラム値の文字数カラム追加関数
#⭐️特徴量追加

def len_col_add(column,change_column,df_train,df_test):
    #⭐️このカラム名変更

    df_train_change_len = df_train[column].str.len().fillna(0)
    df_test_change_len = df_test[column].str.len().fillna(0)

    df_train_change_len_next = pd.DataFrame(df_train_change_len)
    df_test_change_len_next = pd.DataFrame(df_test_change_len)

    #⭐️ここのカラム名変更と変更するカラム名変更
    df_train_change_len_next = df_train_change_len_next.rename(columns = {column:change_column})
    df_test_change_len_next = df_test_change_len_next.rename(columns = {column:change_column})

    #df同士の結合
    df_train = pd.merge(df_train_change_len_next, df_train,left_index=True, right_index=True)
    df_test = pd.merge(df_test_change_len_next, df_test,left_index=True, right_index=True)
    
    return df_train,df_test
df_train,df_test = len_col_add('emp_title','emp_title_len',df_train,df_test)
df_train,df_test = len_col_add('title','title_len',df_train,df_test)
df_train,df_test = len_col_add('home_ownership','home_ownership_len',df_train,df_test)
df_train,df_test = len_col_add('purpose','purpose_len',df_train,df_test)
#df_train['earliest_cr_line'].str[-4:]
df_train_earliest = pd.merge(df_train['earliest_cr_line'].str[-4:], df_train,left_index=True, right_index=True)
df_train_earliest = df_train_earliest.rename(columns = {'earliest_cr_line_x':'earliest_cr_line_year'})
df_train_earliest = pd.merge(df_train['earliest_cr_line'].str[:3], df_train_earliest,left_index=True, right_index=True)
df_train_earliest = df_train_earliest.rename(columns = {'earliest_cr_line':'earliest_cr_line_month'})
df_train_earliest = df_train_earliest.rename(columns = {'earliest_cr_line_y':'earliest_cr_line'})

#df_test['earliest_cr_line'].str[-4:]
df_test_earliest = pd.merge(df_test['earliest_cr_line'].str[-4:], df_test,left_index=True, right_index=True)
df_test_earliest = df_test_earliest.rename(columns = {'earliest_cr_line_x':'earliest_cr_line_year'})
df_test_earliest = pd.merge(df_test['earliest_cr_line'].str[:3], df_test_earliest,left_index=True, right_index=True)
df_test_earliest = df_test_earliest.rename(columns = {'earliest_cr_line':'earliest_cr_line_month'})
df_test_earliest = df_test_earliest.rename(columns = {'earliest_cr_line_y':'earliest_cr_line'})

df_train = df_train_earliest
df_test =  df_test_earliest
#nanを０埋め、intに型変換
df_train['earliest_cr_line_year'] = df_train['earliest_cr_line_year'].fillna(-9999)
df_train['earliest_cr_line_year'] = df_train['earliest_cr_line_year'].astype(int)

df_test['earliest_cr_line_year'] = df_test['earliest_cr_line_year'].fillna(-9999)
df_test['earliest_cr_line_year'] = df_test['earliest_cr_line_year'].astype(int)
df_train['earliest_cr_line_month'] = df_train['earliest_cr_line_month'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug': 8,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12})
df_test['earliest_cr_line_month'] = df_test['earliest_cr_line_month'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug': 8,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12})
df_train['earliest_cr_line_month'].unique()
#nanを０埋め、intに型変換
df_train['earliest_cr_line_month'] = df_train['earliest_cr_line_month'].fillna(-9999)
df_train['earliest_cr_line_month'] = df_train['earliest_cr_line_month'].astype(int)

df_test['earliest_cr_line_month'] = df_test['earliest_cr_line_month'].fillna(-9999)
df_test['earliest_cr_line_month'] = df_test['earliest_cr_line_month'].astype(int)
df_train['duration'] = df_train['issue_d_year'] - df_train['earliest_cr_line_year'] 
df_test['duration'] = df_test['issue_d_year'] - df_test['earliest_cr_line_year'] 
#期間が 100年以上担っているものを変換
df_train.loc[df_train['duration'] > 100, 'duration'] = -9999
df_test.loc[df_test['duration'] > 100, 'duration'] = -9999
#⭐️
#emp_length を数値として切り出す

#タイプ変換する前に欠損値埋め
df_train_emp_len = df_train['emp_length'].fillna(-9999)
df_test_emp_len = df_test['emp_length'].fillna(-9999)

df_train_emp_len_next = pd.DataFrame(df_train_emp_len)
df_test_emp_len_next = pd.DataFrame(df_test_emp_len)

df_train_emp_len_next = df_train_emp_len_next.rename(columns = {'emp_length':'emp_length_num'})
df_test_emp_len_next = df_test_emp_len_next.rename(columns = {'emp_length':'emp_length_num'})

#emp_lengthを数値化
df_train_emp_len_next['emp_length_num'] = df_train_emp_len_next['emp_length_num'].replace({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years': 7,'8 years': 8,'9 years': 9,'10+ years': 10})
df_test_emp_len_next['emp_length_num'] = df_test_emp_len_next['emp_length_num'].replace({'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years': 7,'8 years': 8,'9 years': 9,'10+ years': 10})

df_train_emp_len_next['emp_length_num'] = df_train_emp_len_next['emp_length_num'].astype(int)
df_test_emp_len_next['emp_length_num'] = df_test_emp_len_next['emp_length_num'].astype(int)

#⭐️df同士の結合
df_train = pd.merge(df_train_emp_len_next, df_train,left_index=True, right_index=True)
df_test = pd.merge(df_test_emp_len_next, df_test,left_index=True, right_index=True)

#⭐️
#zip_codeを数値カラムに変換してみる

df_train = pd.merge(df_train['zip_code'].str[:3], df_train,left_index=True, right_index=True)
df_train['zip_code_x'] = df_train['zip_code_x'].astype(int)

df_test = pd.merge(df_test['zip_code'].str[:3], df_test,left_index=True, right_index=True)
df_test['zip_code_x'] = df_test['zip_code_x'].astype(int)


#⭐️
#zip_codeをさらに分割してみる

df_train = pd.merge(df_train['zip_code_y'].str[:1], df_train,left_index=True, right_index=True)
#df_train['zip_code_x'] = df_train['zip_code_x'].astype(int)

df_test = pd.merge(df_test['zip_code_y'].str[:1], df_test,left_index=True, right_index=True)
#df_test['zip_code_x'] = df_test['zip_code_x'].astype(int)

# sample submissionを読み込んで、予測値を代入の後、保存する

#submission = pd.read_csv('sample_submission.csv', index_col=0)
gdp_df = pd.read_csv('../input/homework-for-students4plus/US_GDP_by_State.csv', index_col=0)

desc_df = pd.read_csv('../input/homework-for-students4plus/description.csv', index_col=0)
zip_df = pd.read_csv('../input/homework-for-students4plus/free-zipcode-database.csv', index_col=0)
spi_df = pd.read_csv('../input/homework-for-students4plus/spi.csv', index_col=0)
state_df = pd.read_csv('../input/homework-for-students4plus/statelatlong.csv', index_col=0)

### データを読み込む
# local
#df_train = pd.read_csv('train.csv', index_col=0)
X_test = df_test

#サンプル初期（日付の文字列issue_d を date型に変換している）
#df_train = pd.read_csv('train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)
#X_test = pd.read_csv('test.csv', index_col=0, parse_dates=['issue_d'])


#df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'])
y_train = df_train.loan_condition
X_train = df_train.drop(['loan_condition'], axis=1)


#⭐️最終的には余計なdfは消す
#del df_train
gc.collect()
#色々追加してみる
cat = []
num = []
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        cat.append(col)
    else:
        num.append(col)

        
#数値系標準化正規化
#標準化あれこれ
#対数変換
X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)
X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)

#⭐️
target = 'loan_condition'
X_temp = pd.concat([X_train, y_train], axis=1)

for col in cat:

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

#最終的い中央値埋め
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)

import random
#ランダムに色々作成
clf_arry = []
skf_arry = []

for i in range(0,25):
    rand_int =random.randint(1, 10000)
    
    #random_stateに乱数設定
    clf_random = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,
                                    importance_type='split', learning_rate=0.05, max_depth=-1,
                                    min_child_samples=20, min_child_weight=0.005, min_split_gain=0.0,
                                    n_estimators=99999, n_jobs=-1, num_leaves=15, objective=None,
                                    random_state= rand_int, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                    subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    
    skf_random = StratifiedKFold(n_splits=5, random_state= rand_int, shuffle=True)
    
    print(clf_random)
    print(skf_random)
    
    clf_arry.append(clf_random)
    skf_arry.append(skf_random)
    
print(len(clf_arry))

#⭐️複数検定　複数バージョン


scores = []
y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#------------------------------------
for y in range(0,len(clf_arry)):
    clf = clf_arry[y]
    skf = skf_arry[y]
    
    print(clf)
    print(skf)
    #############################
    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):
        
        print('⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️')
        print(y)
        print(i)
        print('⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️')
        
        #X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
        X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]
        #X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix]
        X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix]
        #clf = LGBMClassifier(**params)
        

        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        #⭐️early_stopping_rounds=200
        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        #⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️
        clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]
        scores.append(roc_auc_score(y_val, y_pred))          


        y_pred_test += clf.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく scores = np.array(scores)
        #print('Ave. CV score is %f' % scores.mean()) 

        #############################
        #print('途中スコア:'')
        #print(scores)
#------------------------------------
y_pred_test

# 最後にfold数で割る
print(len(clf_arry) )
 
y_pred = y_pred_test / (len(clf_arry) * 5)

print('----------------------')
print('各スコア：')
x = 0
for i in scores:
    x += i
    print(i)  
    
print('----------------------')
print(x)
print('Ave. CV score：')
print(x/(len(clf_arry) * 5))
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)
imp
fig, ax = plt.subplots(figsize=(7, 8))
lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# sample submissionを読み込んで、予測値を代入の後、保存する

#submission = pd.read_csv('sample_submission.csv', index_col=0)
submission = pd.read_csv('/kaggle/input/homework-for-students4plus/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred
submission.to_csv('submission.csv')





