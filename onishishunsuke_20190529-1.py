# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

import category_encoders as ce



from sklearn.preprocessing import StandardScaler



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os





# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# 説明変数とターゲットを分ける

X_train = df_train.drop(['loan_condition', 'ID','emp_title','title'], axis=1)

y_train = df_train['loan_condition'] #ターゲット



# テストデータ

X_test = df_test.drop(['ID','emp_title','title'], axis=1) # x_trainと同じ項目（loan_condition以外）を削除



#submit用データ

submit = pd.read_csv('../input/sample_submission.csv', index_col=0) #サンプルファイル読み込み

submit_gb = submit.copy()

submit_lgb = submit.copy()
# GDP付与

df_statelong = pd.read_csv('../input/statelatlong.csv')

df_gdp = pd.read_csv('../input/US_GDP_by_State.csv')



df_gdp = df_gdp[df_gdp.year == 2015] #とりあえず2015年固定にしてしまう

df_gdp = df_gdp.drop(['year'], axis=1)



# 州（long_name⇒short_name）のためにstatelatlong.csvとGDPをjoin

df_state = pd.merge(df_statelong, df_gdp, left_on='City', right_on='State', how='left')

df_state = df_state.drop(['City', 'State_y', 'Latitude', 'Longitude'], axis=1)

#df_state.head()



# 学習データに付与

X_train = pd.merge(X_train, df_state, left_on='addr_state', right_on='State_x', how='left')

X_train = X_train.drop(['State_x'],axis=1)



X_test = pd.merge(X_test, df_state, left_on='addr_state', right_on='State_x', how='left')

X_test = X_test.drop(['State_x'],axis=1)
# S&P付与

df_spi = pd.read_csv('../input/spi.csv')

df_spi['year'] = df_spi['date'].str[-2:]

df_spi['month'] = df_spi['date'].str[-6:].str[:3]

df_spi = df_spi.groupby(['year','month']).mean()



#df_spi.year.value_counts()

X_train['issue_d_year'] = X_train['issue_d'].str[-2:]

X_train['issue_d_month'] = X_train['issue_d'].str[:3]

X_train = pd.merge(X_train, df_spi, left_on=['issue_d_year','issue_d_month'], right_on=['year','month'], how='left')

X_train = X_train.drop(['issue_d','issue_d_year'],axis=1)



X_test['issue_d_year'] = X_test['issue_d'].str[-2:]

X_test['issue_d_month'] = X_test['issue_d'].str[:3]

X_test = pd.merge(X_test, df_spi, left_on=['issue_d_year','issue_d_month'], right_on=['year','month'], how='left')

X_test = X_test.drop(['issue_d','issue_d_year'],axis=1)

# NaNの対応

# 数値カラム

# 学習データ＋テストデータの全体でカウントしてみる。

X_all = pd.concat([X_train, X_test])



X_train.annual_inc.fillna(X_all.annual_inc.median(),inplace=True)

X_train.dti.fillna(X_all.dti.median(),inplace=True)

X_train.delinq_2yrs.fillna(-99,inplace=True)

#X_train.delinq_2yrs.fillna(X_all.delinq_2yrs.median(),inplace=True)

X_train.inq_last_6mths.fillna(-99,inplace=True)

#X_train.inq_last_6mths.fillna(X_all.inq_last_6mths.median(),inplace=True)

X_train.mths_since_last_delinq.fillna(-99,inplace=True)

#X_train.mths_since_last_delinq.fillna(X_all.mths_since_last_delinq.median(),inplace=True)

X_train.mths_since_last_record.fillna(X_all.mths_since_last_record.median(),inplace=True)

X_train.open_acc.fillna(-99,inplace=True)

#X_train.open_acc.fillna(X_all.open_acc.median(),inplace=True)

X_train.pub_rec.fillna(-99,inplace=True)

#X_train.pub_rec.fillna(X_all.pub_rec.median(),inplace=True)

X_train.revol_util.fillna(-99,inplace=True)

#X_train.revol_util.fillna(X_all.revol_util.median(),inplace=True)

X_train.total_acc.fillna(-99,inplace=True)

#X_train.total_acc.fillna(X_all.total_acc.median(),inplace=True)

X_train.collections_12_mths_ex_med.fillna(X_all.collections_12_mths_ex_med.median(),inplace=True)

X_train.mths_since_last_major_derog.fillna(-99,inplace=True)

#X_train.mths_since_last_major_derog.fillna(X_all.mths_since_last_major_derog.median(),inplace=True)

X_train.acc_now_delinq.fillna(-99,inplace=True)

#X_train.acc_now_delinq.fillna(X_all.acc_now_delinq.median(),inplace=True)

X_train.tot_coll_amt.fillna(-99,inplace=True)

#X_train.tot_coll_amt.fillna(X_all.tot_coll_amt.median(),inplace=True)

X_train.tot_cur_bal.fillna(X_all.tot_cur_bal.median(),inplace=True)



X_test.annual_inc.fillna(X_all.annual_inc.median(),inplace=True)

X_test.dti.fillna(X_train.dti.median(),inplace=True)

X_test.delinq_2yrs.fillna(-99,inplace=True)

#X_test.delinq_2yrs.fillna(X_all.delinq_2yrs.median(),inplace=True)

X_test.inq_last_6mths.fillna(-99,inplace=True)

#X_test.inq_last_6mths.fillna(X_all.inq_last_6mths.median(),inplace=True)

X_test.mths_since_last_delinq.fillna(-99,inplace=True)

#X_test.mths_since_last_delinq.fillna(X_all.mths_since_last_delinq.median(),inplace=True)

X_test.mths_since_last_record.fillna(X_all.mths_since_last_record.median(),inplace=True)

X_test.open_acc.fillna(-99,inplace=True)

#X_test.open_acc.fillna(X_all.open_acc.median(),inplace=True)

X_test.pub_rec.fillna(-99,inplace=True)

#X_test.pub_rec.fillna(X_all.pub_rec.median(),inplace=True)

X_test.revol_util.fillna(-99,inplace=True)

#X_test.revol_util.fillna(X_all.revol_util.median(),inplace=True)

X_test.total_acc.fillna(-99,inplace=True)

#X_test.total_acc.fillna(X_all.total_acc.median(),inplace=True)

X_test.collections_12_mths_ex_med.fillna(X_all.collections_12_mths_ex_med.median(),inplace=True)

X_test.mths_since_last_major_derog.fillna(-99,inplace=True)

#X_test.mths_since_last_major_derog.fillna(X_all.mths_since_last_major_derog.median(),inplace=True)

X_test.acc_now_delinq.fillna(-99,inplace=True)

#X_test.acc_now_delinq.fillna(X_all.acc_now_delinq.median(),inplace=True)

X_test.tot_coll_amt.fillna(-99,inplace=True)

#X_test.tot_coll_amt.fillna(X_all.tot_coll_amt.median(),inplace=True)

X_test.tot_cur_bal.fillna(X_all.tot_cur_bal.median(),inplace=True)



# 文字カラム

X_train.emp_length.fillna('#',inplace=True)

#X_train.emp_title.fillna('#',inplace=True)

#X_train.title.fillna('#',inplace=True)

X_train.earliest_cr_line.fillna('#',inplace=True)



X_test.emp_length.fillna('#',inplace=True)

#X_test.emp_title.fillna('#',inplace=True)

#X_test.title.fillna('#',inplace=True)

X_test.earliest_cr_line.fillna('#',inplace=True)
# earliest_cr_lineをyearだけにする。（unique数多いと処理しづらいので、とりあえず）

X_train['earliest_cr_line'] = X_train['earliest_cr_line'].str[-4:]



# 学習データ＋テストデータの全体でカウントしてみる。

#X_all = pd.concat([X_train, X_test])



#emp_title_sum = X_all.emp_title.value_counts()

#X_train.emp_title = X_train.emp_title.map(emp_title_sum)

#X_test.emp_title = X_test.emp_title.map(emp_title_sum)



#title_sum = X_all.title.value_counts()

#X_train.title = X_train.title.map(title_sum)

#X_test.title = X_test.title.map(title_sum)

# feature engineering



X_train['loan_amnt_annual_inc'] = (X_train['loan_amnt']+1) / (X_train['annual_inc']+1) # loan_amnt_annual_inc

X_train['loan_amnt_installment'] = (X_train['loan_amnt']+1) / (X_train['installment']+1) # loan_amnt_installment

X_train['loan_amnt_tot_cur_bal'] = (X_train['loan_amnt']+1) / (X_train['tot_cur_bal']+1) # loan_amnt_tot_cur_bal

X_train['installment_annual_inc'] = (X_train['installment']+1) / (X_train['annual_inc']+1)  # installment_annual_inc

X_train['revol_bal_annual_inc'] = (X_train['revol_bal']+1) / (X_train['annual_inc']+1)  # revol_bal_annual_inc

X_train['revol_bal_installment'] = (X_train['revol_bal']+1) / (X_train['installment']+1)  # revol_bal_installment

X_train['loan_amnt_revol_bal_annual_inc'] = (X_train['loan_amnt'] + X_train['revol_bal'] + 1) / (X_train['annual_inc']+1)  # loan_amnt_revol_bal_annual_inc

X_train['annual_inc_tot_cur_bal'] = (X_train['annual_inc']+1) / (X_train['tot_cur_bal']+1) # annual_inc_tot_cur_bal

X_train['installment_tot_cur_bal'] = (X_train['installment']+1) / (X_train['tot_cur_bal']+1) # installment_tot_cur_bal

X_train['installment_annual_inc_tot_cur_bal'] = (X_train['installment']+1) / (X_train['annual_inc'] + X_train['tot_cur_bal'] +1)  # installment_annual_inc_tot_cur_bal

X_train['loan_amnt_revol_bal_annual_inc_tot_cur_bal'] = (X_train['loan_amnt'] + X_train['revol_bal']+1) / (X_train['annual_inc'] + X_train['tot_cur_bal'] +1) # loan_amnt_revol_bal_annual_inc_tot_cur_bal

X_train['tot_coll_amt_flg'] = pd.cut(X_train['tot_coll_amt'], [-np.inf,-1,1,np.inf],labels=False) # tot_coll_amt_flg

X_train['close_population'] = X_train['close'] / X_train['Population (million)']



X_test['loan_amnt_annual_inc'] = (X_test['loan_amnt']+1) / (X_test['annual_inc']+1) # loan_amnt_annual_inc

X_test['loan_amnt_installment'] = (X_test['loan_amnt']+1) / (X_test['installment']+1) # loan_amnt_installment

X_test['loan_amnt_tot_cur_bal'] = (X_test['loan_amnt']+1) / (X_test['tot_cur_bal']+1) # loan_amnt_tot_cur_bal

X_test['installment_annual_inc'] = (X_test['installment']+1) / (X_test['annual_inc']+1)  # installment_annual_inc

X_test['revol_bal_annual_inc'] = (X_test['revol_bal']+1) / (X_test['annual_inc']+1)  # revol_bal_annual_inc

X_test['revol_bal_installment'] = (X_test['revol_bal']+1) / (X_test['installment']+1)  # revol_bal_installment

X_test['loan_amnt_revol_bal_annual_inc'] = (X_test['loan_amnt'] + X_test['revol_bal'] + 1) / (X_test['annual_inc']+1)  # loan_amnt_revol_bal_annual_inc

X_test['annual_inc_tot_cur_bal'] = (X_test['annual_inc']+1) / (X_test['tot_cur_bal']+1) # annual_inc_tot_cur_bal

X_test['installment_tot_cur_bal'] = (X_test['installment']+1) / (X_test['tot_cur_bal']+1) # installment_tot_cur_bal

X_test['installment_annual_inc_tot_cur_bal'] = (X_test['installment']+1) / (X_test['annual_inc'] + X_test['tot_cur_bal'] +1)  # installment_annual_inc_tot_cur_bal

X_test['loan_amnt_revol_bal_annual_inc_tot_cur_bal'] = (X_test['loan_amnt'] + X_test['revol_bal']+1) / (X_test['annual_inc'] + X_test['tot_cur_bal'] +1) # loan_amnt_revol_bal_annual_inc_tot_cur_bal

X_test['tot_coll_amt_flg'] = pd.cut(X_test['tot_coll_amt'], [-np.inf,-1,1,np.inf],labels=False) # tot_coll_amt_flg

X_test['close_population'] = X_test['close'] / X_test['Population (million)']
# 数値系のエンコード

X_train.installment = X_train.installment.apply(np.log1p)

X_train.annual_inc = X_train.annual_inc.apply(np.log1p)



X_test.installment = X_test.installment.apply(np.log1p)

X_test.annual_inc = X_test.annual_inc.apply(np.log1p)
X_train['grade'] = X_train['grade'].replace('A',1)

X_train['grade'] = X_train['grade'].replace('B',2)

X_train['grade'] = X_train['grade'].replace('C',3)

X_train['grade'] = X_train['grade'].replace('D',4)

X_train['grade'] = X_train['grade'].replace('E',5)

X_train['grade'] = X_train['grade'].replace('F',6)

X_train['grade'] = X_train['grade'].replace('G',7)



X_test['grade'] = X_test['grade'].replace('A',1)

X_test['grade'] = X_test['grade'].replace('B',2)

X_test['grade'] = X_test['grade'].replace('C',3)

X_test['grade'] = X_test['grade'].replace('D',4)

X_test['grade'] = X_test['grade'].replace('E',5)

X_test['grade'] = X_test['grade'].replace('F',6)

X_test['grade'] = X_test['grade'].replace('G',7)
X_train['sub_grade'] = X_train['sub_grade'].replace('A1',11)

X_train['sub_grade'] = X_train['sub_grade'].replace('A2',12)

X_train['sub_grade'] = X_train['sub_grade'].replace('A3',13)

X_train['sub_grade'] = X_train['sub_grade'].replace('A4',14)

X_train['sub_grade'] = X_train['sub_grade'].replace('A5',15)

X_train['sub_grade'] = X_train['sub_grade'].replace('B1',21)

X_train['sub_grade'] = X_train['sub_grade'].replace('B2',22)

X_train['sub_grade'] = X_train['sub_grade'].replace('B3',23)

X_train['sub_grade'] = X_train['sub_grade'].replace('B4',24)

X_train['sub_grade'] = X_train['sub_grade'].replace('B5',25)

X_train['sub_grade'] = X_train['sub_grade'].replace('C1',31)

X_train['sub_grade'] = X_train['sub_grade'].replace('C2',32)

X_train['sub_grade'] = X_train['sub_grade'].replace('C3',33)

X_train['sub_grade'] = X_train['sub_grade'].replace('C4',34)

X_train['sub_grade'] = X_train['sub_grade'].replace('C5',35)

X_train['sub_grade'] = X_train['sub_grade'].replace('D1',41)

X_train['sub_grade'] = X_train['sub_grade'].replace('D2',42)

X_train['sub_grade'] = X_train['sub_grade'].replace('D3',43)

X_train['sub_grade'] = X_train['sub_grade'].replace('D4',44)

X_train['sub_grade'] = X_train['sub_grade'].replace('D5',45)

X_train['sub_grade'] = X_train['sub_grade'].replace('E1',51)

X_train['sub_grade'] = X_train['sub_grade'].replace('E2',52)

X_train['sub_grade'] = X_train['sub_grade'].replace('E3',53)

X_train['sub_grade'] = X_train['sub_grade'].replace('E4',54)

X_train['sub_grade'] = X_train['sub_grade'].replace('E5',55)

X_train['sub_grade'] = X_train['sub_grade'].replace('F1',61)

X_train['sub_grade'] = X_train['sub_grade'].replace('F2',62)

X_train['sub_grade'] = X_train['sub_grade'].replace('F3',63)

X_train['sub_grade'] = X_train['sub_grade'].replace('F4',64)

X_train['sub_grade'] = X_train['sub_grade'].replace('F5',65)

X_train['sub_grade'] = X_train['sub_grade'].replace('G1',71)

X_train['sub_grade'] = X_train['sub_grade'].replace('G2',72)

X_train['sub_grade'] = X_train['sub_grade'].replace('G3',73)

X_train['sub_grade'] = X_train['sub_grade'].replace('G4',74)

X_train['sub_grade'] = X_train['sub_grade'].replace('G5',75)



X_test['sub_grade'] = X_test['sub_grade'].replace('A1',11)

X_test['sub_grade'] = X_test['sub_grade'].replace('A2',12)

X_test['sub_grade'] = X_test['sub_grade'].replace('A3',13)

X_test['sub_grade'] = X_test['sub_grade'].replace('A4',14)

X_test['sub_grade'] = X_test['sub_grade'].replace('A5',15)

X_test['sub_grade'] = X_test['sub_grade'].replace('B1',21)

X_test['sub_grade'] = X_test['sub_grade'].replace('B2',22)

X_test['sub_grade'] = X_test['sub_grade'].replace('B3',23)

X_test['sub_grade'] = X_test['sub_grade'].replace('B4',24)

X_test['sub_grade'] = X_test['sub_grade'].replace('B5',25)

X_test['sub_grade'] = X_test['sub_grade'].replace('C1',31)

X_test['sub_grade'] = X_test['sub_grade'].replace('C2',32)

X_test['sub_grade'] = X_test['sub_grade'].replace('C3',33)

X_test['sub_grade'] = X_test['sub_grade'].replace('C4',34)

X_test['sub_grade'] = X_test['sub_grade'].replace('C5',35)

X_test['sub_grade'] = X_test['sub_grade'].replace('D1',41)

X_test['sub_grade'] = X_test['sub_grade'].replace('D2',42)

X_test['sub_grade'] = X_test['sub_grade'].replace('D3',43)

X_test['sub_grade'] = X_test['sub_grade'].replace('D4',44)

X_test['sub_grade'] = X_test['sub_grade'].replace('D5',45)

X_test['sub_grade'] = X_test['sub_grade'].replace('E1',51)

X_test['sub_grade'] = X_test['sub_grade'].replace('E2',52)

X_test['sub_grade'] = X_test['sub_grade'].replace('E3',53)

X_test['sub_grade'] = X_test['sub_grade'].replace('E4',54)

X_test['sub_grade'] = X_test['sub_grade'].replace('E5',55)

X_test['sub_grade'] = X_test['sub_grade'].replace('F1',61)

X_test['sub_grade'] = X_test['sub_grade'].replace('F2',62)

X_test['sub_grade'] = X_test['sub_grade'].replace('F3',63)

X_test['sub_grade'] = X_test['sub_grade'].replace('F4',64)

X_test['sub_grade'] = X_test['sub_grade'].replace('F5',65)

X_test['sub_grade'] = X_test['sub_grade'].replace('G1',71)

X_test['sub_grade'] = X_test['sub_grade'].replace('G2',72)

X_test['sub_grade'] = X_test['sub_grade'].replace('G3',73)

X_test['sub_grade'] = X_test['sub_grade'].replace('G4',74)

X_test['sub_grade'] = X_test['sub_grade'].replace('G5',75)

# 文字列抽出

cats = []

for col in X_train.columns:

        if X_train[col].dtype == 'object':

            cats.append(col)

            

            print(col, X_train[col].nunique())
# 文字列のエンコード

oe = ce.OrdinalEncoder(cols=cats,return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.fit_transform(X_test[cats])
X_train['loan_amnt_annual_inc_grade'] = (X_train['loan_amnt_annual_inc']+1) / X_train['grade'] #loan_amnt_annual_inc grade

X_train['loan_amnt_tot_cur_bal_grade'] = (X_train['loan_amnt_tot_cur_bal']+1) / X_train['grade'] #loan_amnt_tot_cur_bal grade

X_train['revol_bal_annual_inc_grade'] = (X_train['revol_bal_annual_inc']+1) / X_train['grade']  #revol_bal_annual_inc grade

X_train['installment_annual_inc_grade'] = (X_train['installment_annual_inc']+1) / X_train['grade'] #installment_annual_inc grade

X_train['addr_state_home_ownership'] = X_train['addr_state'] / X_train['home_ownership'] #addr_state home_ownership

X_train['revol_bal_sub_grade'] = (X_train['revol_bal']+1) / X_train['sub_grade'] #revol_bal sub_grade



X_test['loan_amnt_annual_inc_grade'] = (X_test['loan_amnt_annual_inc']+1) / X_test['grade'] #loan_amnt_annual_inc grade

X_test['loan_amnt_tot_cur_bal_grade'] = (X_test['loan_amnt_tot_cur_bal']+1) / X_test['grade'] #loan_amnt_tot_cur_bal grade

X_test['revol_bal_annual_inc_grade'] = (X_test['revol_bal_annual_inc']+1) / X_test['grade']  #revol_bal_annual_inc grade

X_test['installment_annual_inc_grade'] = (X_test['installment_annual_inc']+1) / X_test['grade'] #installment_annual_inc grade

X_test['addr_state_home_ownership'] = X_test['addr_state'] / X_test['home_ownership'] #addr_state home_ownership

X_test['revol_bal_sub_grade'] = (X_test['revol_bal']+1) / X_test['sub_grade'] #revol_bal sub_grade
X_train.describe()
X_test.describe()
# 正規化

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_train.describe()
X_test.describe()
# GradientBoostingClassifierで全データで再学習

clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する

submit_gb['loan_condition'] = y_pred # 予測値で上書き
#LGB2

import lightgbm as lgb

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt



plt.style.use('ggplot')

%matplotlib inline



scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #scikitlearnのLightGBM

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9, #colsample_bytree：1木あたりで使う項目数（割合）

                         importance_type='split', learning_rate=0.05, max_depth=-1,

                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None, #n_estimators（木の数）は最大にしておいてearlyStoppingに任せる

                         random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                         subsample=0.9, subsample_for_bin=200000, subsample_freq=0) #lightGBMは連続値もbinningする。



    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

   

    y_pred_2 = clf.predict_proba(X_test)[:,1]

    submit['loan_condition'] = y_pred_2

    submit_lgb =  pd.merge(submit_lgb, submit, left_on=['ID'], right_on=['ID'], how='left')



    score = roc_auc_score(y_val, y_pred)

    print(score)



    fig, ax = plt.subplots(figsize=(10, 15))

    lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
submit.head()
submit_gb.head()
submit_lgb.head()
submit = pd.merge(submit_lgb, submit_gb, left_on=['ID'], right_on=['ID'], how='left')

#submit['loan_condition'] = (submit['loan_condition_x'] + submit['loan_condition_y'])/2

#submit = submit.drop(['loan_condition_x','loan_condition_y'],axis=1)

submit.columns = ['a','b','c','d','e','f','g']

submit['loan_condition'] = (submit['b']+submit['c']+submit['d']+submit['e']+submit['f']+submit['g'])/6

submit = submit.drop(['a','b','c','d','e','f','g'],axis=1)
submit.head()
submit.to_csv('submission.csv')