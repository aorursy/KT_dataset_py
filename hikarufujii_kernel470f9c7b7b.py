import gc
import tensorflow as tf
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
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)#, skiprows=lambda x: x%20!=0)

#df_test = #testデータの読み込みをtrainを参考に書いて見ましょう！

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0)#, skiprows=lambda x: x%20!=0)
# DataFrameのshapeで行数と列数を確認してみましょう。

df_train.shape, df_test.shape
# 先頭5行をみてみます。

#df_train.head()
#df_test.head()
#df_train[(df_train.loan_amnt>1000)&&(df_train.grade =='A')]
#df_train[df_train.loan_condition==1]
#df_train[df_train.loan_condition==1].loan_amnt.mean()
# 上の貸し倒れたローンに対するものを参考に、貸し倒れていないローンの平均額を算出みてください。

#df_train[df_train.loan_condition==0].loan_amnt.mean()
y_train = df_train.loan_condition

#loan_condition削除

X_train = df_train.drop(['loan_condition'], axis =1)



X_test = df_test
##特徴量追加
#返済回数

X_train["payment_count"]=X_train["loan_amnt"]/X_train["installment"]

X_test["payment_count"]=X_test["loan_amnt"]/X_test["installment"]
# クレジット利用開始年月が入っていない場合、Issue_dをで補完

X_train['earliest_cr_line'].fillna(X_train["issue_d"],inplace=True)

X_test['earliest_cr_line'].fillna(X_test["issue_d"],inplace=True)



#最初に利用してからの日数

X_train["credit_hist_days"]=(pd.to_datetime(X_train["issue_d"])-pd.to_datetime(X_train['earliest_cr_line'])).apply(lambda x: x.days)

X_test["credit_hist_days"]=(pd.to_datetime(X_test["issue_d"])-pd.to_datetime(X_test['earliest_cr_line'])).apply(lambda x: x.days)
#tot_cur_bal_avg統計量追加

X_train["tot_cur_bal_avg_ratio"]=X_train["tot_cur_bal"]/X_train["tot_cur_bal"].mean()

X_test["tot_cur_bal_avg_ratio"]=X_train["tot_cur_bal"]/X_test["tot_cur_bal"].mean()
X_train
from sklearn.preprocessing import StandardScaler , MinMaxScaler
#scaler = StandardScaler()

scaler = MinMaxScaler()
##別々にフィットする

#X_train['loan_amnt'] = scaler.fit_transform(X_train[['loan_amnt']])

#X_test['loan_amnt'] = scaler.transform(X_test[['loan_amnt']])



##Train と　Testを一緒にフィットする 

scaler.fit(pd.concat([X_train['annual_inc'],X_test['annual_inc']]).to_frame())

X_train['annual_inc']  = scaler.transform(X_train[['annual_inc']])

X_test['annual_inc']  = scaler.transform(X_test[['annual_inc']])     
##Train と　Testを一緒にフィットする 

scaler.fit(pd.concat([X_train['installment'],X_test['installment']]).to_frame())

X_train['installment']  = scaler.transform(X_train[['installment']])

X_test['installment']  = scaler.transform(X_test[['installment']])     
#plt.figure(figsize=[7,7])

#X_train.annual_inc.hist(bins=100)

#X_test.annual_inc.hist(bins=20)

#plt.show
#plt.figure(figsize=[7,7])

#np.log1p(X_train.annual_inc).hist(bins=100)

#np.log1p(X_test.annual_inc).hist(bins=10)

#plt.show
#plt.figure(figsize=[7,7])

#X_train.annual_inc.apply(np.log).hist(bins=100)

#np.log1p(X_train.annual_inc).hist(bins=100)

#X_train.annual_inc.apply(np.log).hist(bins=100)

#plt.show
##給与を対数変換してます

#scaler.fit(pd.concat([X_train['annual_inc'],X_test['annual_inc']]).to_frame())

#X_train['annual_inc']  = scaler.transform(X_train[['annual_inc']])

#X_test['annual_inc']  = scaler.transform(X_test[['annual_inc']])     

X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
X_train['emp_title'].head(10) # カテゴリよりテキストとして扱ったほうが良いかもしれない
cats
#col = 'purpose'



#encoder = OneHotEncoder()

#enc_train = encoder.fit_transform(X_train[col].values)

#enc_test = encoder.transform(X_test[col].values)
#enc_train.head()
#enc_test.head()
#X_train.head()
#X_train =pd.concat([X_train, enc_train], axis=1)
#X_train.head()
# Onehotの例を参考にやってみましょう

# https://contrib.scikit-learn.org/categorical-encoding/ordinal.html

#encoder = OrdinalEncoder()

#enc_train = encoder.fit_transform(X_train[col].values)

#enc_test = encoder.transform(X_test[col].values)

#col = 'purpose'

#value_couontsで集計した結果を、

#summary = X_train[col].value_counts()

#summary
# mapする。

#enc_train = X_train[col].map(summary)

#enc_test = X_test[col].map(summary)
#Purposeのエンコーディング

col = 'purpose'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴu8リ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
X_train = pd.concat([X_train.drop(col, axis=1), enc_train], axis = 1)

X_test = pd.concat([X_test.drop(col, axis=1), enc_test], axis = 1)
X_train = X_train.rename(columns={0: 'purpose2'})

X_test = X_test.rename(columns={'purpose': 'purpose2'})
X_test.head()
#Gradeのエンコーディング

col = 'grade'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴu8リ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
X_train = pd.concat([X_train.drop(col, axis=1), enc_train], axis = 1)

X_test = pd.concat([X_test.drop(col, axis=1), enc_test], axis = 1)
X_train = X_train.rename(columns={0: 'grade2'})

X_test = X_test.rename(columns={0: 'grade2'})
#Sub Gradeのエンコーディング

col = 'sub_grade'

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴu8リ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
X_train = pd.concat([X_train.drop(col, axis=1), enc_train], axis = 1)

X_test = pd.concat([X_test.drop(col, axis=1), enc_test], axis = 1)
X_train = X_train.rename(columns={0: 'sub_grade2'})

X_test = X_test.rename(columns={'sub_grade': 'sub_grade2'})
##titleの内容修正  ２、３、４、５、６を試したが、３項目変更がもっともAUC高い

X_train.loc[X_train['title'] =='Debt Consolidation','title'] = 'Debt consolidation'

X_train.loc[X_train['title'] == 'consolidation', 'title'] = 'Debt consolidation'

X_train.loc[X_train['title'] == 'debt consolidation', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'Consolidation', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'Debt Consolidation Loan', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'Consolidation Loan', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'Consolidate', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'consolidate', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'DEBT CONSOLIDATION', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'debt consolidation loan', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'Debt consolidation loan', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'consolidation loan', 'title'] = 'Debt consolidation'

#X_train.loc[X_train['title'] == 'debt_consolidation', 'title'] = 'Debt consolidation'







X_test.loc[X_test['title'] =='Debt Consolidation','title'] = 'Debt consolidation'

X_test.loc[X_test['title'] == 'consolidation', 'title'] = 'Debt consolidation'

X_test.loc[X_test['title'] == 'debt consolidation', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'Consolidation', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'Debt Consolidation Loan', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'Consolidation Loan', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'Consolidate', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'consolidate', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'DEBT CONSOLIDATION', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'debt consolidation loan', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'Debt consolidation loan', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'consolidation loan', 'title'] = 'Debt consolidation'

#X_test.loc[X_test['title'] == 'debt_consolidation', 'title'] = 'Debt consolidation'
u = X_train['emp_length'].unique()

print(u)

print(type(u))
col = 'emp_length'

# value_couontsで集計した結果を、

summary = X_train[col].value_counts()

summary
#勤続年数を数値に変換するテーブル作成

emp_leng_transfer = pd.DataFrame({'emp_length' : u,

                                                                  'emp_years' : [10,2,1,3,0,7,5,6,8,0.5,4,9]})
X_train = X_train.merge(emp_leng_transfer, on='emp_length', how ='left')

X_train = X_train.drop(['emp_length'], axis =1)

X_test = X_test.merge(emp_leng_transfer, on='emp_length', how ='left')

X_test = X_test.drop(['emp_length'], axis =1)
col = 'emp_years'

# value_couontsで集計した結果を、

summary = X_train[col].value_counts()

summary
df_statelatlong = pd.read_csv('../input/homework-for-students2/statelatlong.csv')

df_US_GDP_by_State = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')

#名前変更

df_statelatlong = df_statelatlong.rename(columns={'State': 'addr_state'})

#名前変更

df_US_GDP_by_State = df_US_GDP_by_State.rename(columns={'State': 'City'})
#マスタの結合

df_statelatlong = pd.concat([df_statelatlong.drop(['City'], axis=1), df_US_GDP_by_State], axis = 1,join='inner')
#statelatlongを結合

#X_train = pd.concat([X_train.drop(['addr_state'], axis=1), df_statelatlong], axis = 1,join='inner')

#X_test = pd.concat([X_test.drop(['addr_state'], axis=1), df_statelatlong], axis = 1,join='inner')



X_train=pd.merge(X_train, df_statelatlong, on = 'addr_state', how ='left')

X_test=pd.merge(X_test, df_statelatlong, on = 'addr_state', how ='left')



X_train = X_train.drop(['City'], axis=1)

X_train = X_train.drop(['year'], axis=1)

X_train = X_train.drop(['addr_state'], axis=1)



X_test = X_test.drop(['City'], axis=1)

X_test = X_test.drop(['year'], axis=1)

X_test = X_test.drop(['addr_state'], axis=1)

TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



cats.remove('emp_title')
#不要列の削除　

#titleはPurposeとおなじなので、削除

#X_train = X_train.drop(['title'],axis=1)

#X_test = X_test.drop(['title'],axis=1)

#zip_codeも削除

#X_train = X_train.drop(['zip_code'],axis=1)

#X_test = X_test.drop(['zip_code'],axis=1)



#ローン開始時期はテストと違うので削除

#X_train = X_train.drop(['issue_d'],axis=1)

#X_test = X_test.drop(['issue_d'],axis=1)



#cats.remove('issue_d')

cats.remove('purpose')

cats.remove('emp_length')

cats.remove('addr_state')

cats.remove('grade')

cats.remove('sub_grade')

#cats.remove('zip_code')

#cats.remove('title')

cats
 # 自分で書いてみましょう

encoder = OrdinalEncoder(cols=cats)
X_train[cats]=encoder.fit_transform(X_train[cats])

X_test[cats]=encoder.transform(X_test[cats])
#X_train['titlenull_flg']=X_train.emp_title.isnull().astype(int)

#X_test['titlenull_flg']=X_test.emp_title.isnull().astype(int)
#X_train['titlenull_flg'].value_counts()
##勤続年数

X_train['titlenull_flg']=X_train['emp_years'].apply(lambda x : 1 if x == 0 else 0)

X_test['titlenull_flg']=X_test['emp_years'].apply(lambda x : 1 if x == 0 else 0)
X_train['titlenull_flg'].value_counts()
# 以下を参考に自分で書いてみましょう 

X_train.drop(['emp_title'],axis = 1, inplace=True)

X_test.drop(['emp_title'],axis = 1, inplace=True)



X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
TXT_train.fillna('#', inplace=True) #inplaceは上書き

TXT_test.fillna('#', inplace=True)
tfidf = TfidfVectorizer(max_features=1000, use_idf=True) #カラム方向に正規化する
tfidf
TXT_train = tfidf.fit_transform(TXT_train)

TXT_test = tfidf.transform(TXT_test)
#疎行列が帰ってきます。

TXT_train
TXT_train.shape
# todenseで密行列に変換できますが、ほどんどゼロであることがみて取れます。

TXT_train.todense()
TXT_train
# Return a Coordinate (coo) representation of the Compresses-Sparse-Column (csc) matrix.

coo_train = TXT_train.tocoo(copy=False)

coo_test = TXT_test.tocoo(copy=False)
coo_train
#DataFrame形式に値を取り出し

df_txt_train=pd.DataFrame({'index': coo_train.row, 'col': coo_train.col, 'data': coo_train.data}

                 )[['index', 'col', 'data']].sort_values(['index', 'col']

                 ).reset_index(drop=True)

df_txt_test=pd.DataFrame({'index': coo_test.row, 'col': coo_test.col, 'data': coo_test.data}

                 )[['index', 'col', 'data']].sort_values(['index', 'col']

                 ).reset_index(drop=True)
#df_txt_train.info()
#df_txt_test.info()
#Mergeの際にキーとなるndexが複数あるので、Group　by　それ以外の値は最小値をとる。。。。。これでいいのか！？

df_txt_train=df_txt_train.groupby(['index'], as_index=False)['col','data'].min()

df_txt_test=df_txt_test.groupby(['index'], as_index=False)['col','data'].min()
#index列の追加

X_train['index'] = X_train.index

X_test['index'] = X_test.index
##X_train = pd.concat([X_train, df_txt_train], axis = 1)

X_train = pd.merge(X_train, df_txt_train, on = 'index', how ='outer')

X_test = pd.merge(X_test, df_txt_test, on = 'index', how ='outer')
#indexは削除

X_train = X_train.drop(['index'],axis=1)

X_test = X_test.drop(['index'],axis=1)
X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)
X_train
import lightgbm as lgb

from lightgbm import LGBMClassifier



# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []



skf = StratifiedKFold(n_splits=6, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

#clf = GradientBoostingClassifier()

    #デフォルト　LGBM

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                               importance_type='split', learning_rate=0.05, max_depth=-1,

                              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                               n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                               random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    #clf.fit(X_train_, y_train_)

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
# 

print(np.mean(scores))

print(scores)
# Scaller issue_d

print(np.mean(scores))

print(scores)
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1]
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)#, skiprows=lambda x: x%20!=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
submission.head()