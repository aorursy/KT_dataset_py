# ライブラリのインポート

import numpy as np

import pandas as pd

import category_encoders as ce



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm



#データ読込

df_train = []

df_test = []

df_train_small = pd.read_csv('../input/train_small.csv',index_col=0)

df_train = pd.read_csv('../input/train.csv',index_col=0)

df_test = pd.read_csv('../input/test.csv',index_col=0)

df_spi = pd.read_csv('../input/spi.csv')

df_statelatlong = pd.read_csv('../input/statelatlong.csv')

df_US_GDP_by_State = pd.read_csv('../input/US_GDP_by_State.csv')

#テーブル結合

df_train = df_train.reset_index().merge(df_statelatlong.rename(columns={"State":"addr_state"})).set_index(df_train.index.names)

df_test = df_test.reset_index().merge(df_statelatlong.rename(columns={"State":"addr_state"})).set_index(df_test.index.names)



#df_train = df_train.reset_index().merge(df_US_GDP_by_State.rename(columns={"State":"City"})).set_index(df_train.index.names)

#df_test = df_test.reset_index().merge(df_US_GDP_by_State.rename(columns={"State":"City"})).set_index(df_test.index.names)

y_train = df_train.loan_condition

x_train = df_train.drop(['loan_condition', 'issue_d'], axis=1)

x_test = df_test.drop(['issue_d'], axis=1)
cats = []

for col in x_train.columns:

    if x_train[col].dtype == 'object':

        cats.append(col)

        print(col,x_train[col].nunique())
cats.remove('emp_title')

cats
x_train.grade.value_counts()
x_train.grade = x_train.grade.replace('B', 254553)

x_train.grade = x_train.grade.replace('C', 245881)

x_train.grade = x_train.grade.replace('A', 148207)

x_train.grade = x_train.grade.replace('D', 139554)

x_train.grade = x_train.grade.replace('E', 70707)

x_train.grade = x_train.grade.replace('F', 23049)

x_train.grade = x_train.grade.replace('G', 5489)



x_test.grade = x_test.grade.replace('B', 254553)

x_test.grade = x_test.grade.replace('C', 245881)

x_test.grade = x_test.grade.replace('A', 148207)

x_test.grade = x_test.grade.replace('D', 139554)

x_test.grade = x_test.grade.replace('E', 70707)

x_test.grade = x_test.grade.replace('F', 23049)

x_test.grade = x_test.grade.replace('G', 5489)



x_train.head()
cats.remove('grade')

cats
x_train.sub_grade.value_counts()
x_train.sub_grade = x_train.sub_grade.replace('B3', 56327)

x_train.sub_grade = x_train.sub_grade.replace('B4', 55629)

x_train.sub_grade = x_train.sub_grade.replace('C1', 53393)

x_train.sub_grade = x_train.sub_grade.replace('C2', 52241)

x_train.sub_grade = x_train.sub_grade.replace('C3', 50167)

x_train.sub_grade = x_train.sub_grade.replace('C4', 48860)

x_train.sub_grade = x_train.sub_grade.replace('B5', 48839)

x_train.sub_grade = x_train.sub_grade.replace('B2', 48783)

x_train.sub_grade = x_train.sub_grade.replace('B1', 44975)

x_train.sub_grade = x_train.sub_grade.replace('A5', 44818)

x_train.sub_grade = x_train.sub_grade.replace('C5', 41220)

x_train.sub_grade = x_train.sub_grade.replace('D1', 36239)

x_train.sub_grade = x_train.sub_grade.replace('A4', 34531)

x_train.sub_grade = x_train.sub_grade.replace('D2', 29804)

x_train.sub_grade = x_train.sub_grade.replace('D3', 26560)

x_train.sub_grade = x_train.sub_grade.replace('D4', 25560)

x_train.sub_grade = x_train.sub_grade.replace('A3', 23458)

x_train.sub_grade = x_train.sub_grade.replace('A1', 22915)

x_train.sub_grade = x_train.sub_grade.replace('A2', 22485)

x_train.sub_grade = x_train.sub_grade.replace('D5', 21391)

x_train.sub_grade = x_train.sub_grade.replace('E1', 18269)

x_train.sub_grade = x_train.sub_grade.replace('E2', 17004)

x_train.sub_grade = x_train.sub_grade.replace('E3', 14135)

x_train.sub_grade = x_train.sub_grade.replace('E4', 11724)

x_train.sub_grade = x_train.sub_grade.replace('E5', 9575)

x_train.sub_grade = x_train.sub_grade.replace('F1', 7219)

x_train.sub_grade = x_train.sub_grade.replace('F2', 5393)

x_train.sub_grade = x_train.sub_grade.replace('F3', 4433)

x_train.sub_grade = x_train.sub_grade.replace('F4', 3410)

x_train.sub_grade = x_train.sub_grade.replace('F5', 2594)

x_train.sub_grade = x_train.sub_grade.replace('G1', 1871)

x_train.sub_grade = x_train.sub_grade.replace('G2', 1398)

x_train.sub_grade = x_train.sub_grade.replace('G3', 981)

x_train.sub_grade = x_train.sub_grade.replace('G4', 663)

x_train.sub_grade = x_train.sub_grade.replace('G5', 576)



x_test.sub_grade = x_test.sub_grade.replace('B3', 56327)

x_test.sub_grade = x_test.sub_grade.replace('B4', 55629)

x_test.sub_grade = x_test.sub_grade.replace('C1', 53393)

x_test.sub_grade = x_test.sub_grade.replace('C2', 52241)

x_test.sub_grade = x_test.sub_grade.replace('C3', 50167)

x_test.sub_grade = x_test.sub_grade.replace('C4', 48860)

x_test.sub_grade = x_test.sub_grade.replace('B5', 48839)

x_test.sub_grade = x_test.sub_grade.replace('B2', 48783)

x_test.sub_grade = x_test.sub_grade.replace('B1', 44975)

x_test.sub_grade = x_test.sub_grade.replace('A5', 44818)

x_test.sub_grade = x_test.sub_grade.replace('C5', 41220)

x_test.sub_grade = x_test.sub_grade.replace('D1', 36239)

x_test.sub_grade = x_test.sub_grade.replace('A4', 34531)

x_test.sub_grade = x_test.sub_grade.replace('D2', 29804)

x_test.sub_grade = x_test.sub_grade.replace('D3', 26560)

x_test.sub_grade = x_test.sub_grade.replace('D4', 25560)

x_test.sub_grade = x_test.sub_grade.replace('A3', 23458)

x_test.sub_grade = x_test.sub_grade.replace('A1', 22915)

x_test.sub_grade = x_test.sub_grade.replace('A2', 22485)

x_test.sub_grade = x_test.sub_grade.replace('D5', 21391)

x_test.sub_grade = x_test.sub_grade.replace('E1', 18269)

x_test.sub_grade = x_test.sub_grade.replace('E2', 17004)

x_test.sub_grade = x_test.sub_grade.replace('E3', 14135)

x_test.sub_grade = x_test.sub_grade.replace('E4', 11724)

x_test.sub_grade = x_test.sub_grade.replace('E5', 9575)

x_test.sub_grade = x_test.sub_grade.replace('F1', 7219)

x_test.sub_grade = x_test.sub_grade.replace('F2', 5393)

x_test.sub_grade = x_test.sub_grade.replace('F3', 4433)

x_test.sub_grade = x_test.sub_grade.replace('F4', 3410)

x_test.sub_grade = x_test.sub_grade.replace('F5', 2594)

x_test.sub_grade = x_test.sub_grade.replace('G1', 1871)

x_test.sub_grade = x_test.sub_grade.replace('G2', 1398)

x_test.sub_grade = x_test.sub_grade.replace('G3', 981)

x_test.sub_grade = x_test.sub_grade.replace('G4', 663)

x_test.sub_grade = x_test.sub_grade.replace('G5', 576)



x_train.head()
cats.remove('sub_grade')

cats
x_train.emp_length.value_counts()
x_train.emp_length = x_train.emp_length.replace('10+ years', 10)

x_train.emp_length = x_train.emp_length.replace('2 years', 2)

x_train.emp_length = x_train.emp_length.replace('< 1 year', 0.5)

x_train.emp_length = x_train.emp_length.replace('3 years', 3)

x_train.emp_length = x_train.emp_length.replace('1 year', 1)

x_train.emp_length = x_train.emp_length.replace('5 years', 5)

x_train.emp_length = x_train.emp_length.replace('4 years', 4)

x_train.emp_length = x_train.emp_length.replace('7 years', 7)

x_train.emp_length = x_train.emp_length.replace('8 years', 8)

x_train.emp_length = x_train.emp_length.replace('6 years', 6)

x_train.emp_length = x_train.emp_length.replace('9 years', 9)



x_test.emp_length = x_test.emp_length.replace('10+ years', 10)

x_test.emp_length = x_test.emp_length.replace('2 years', 2)

x_test.emp_length = x_test.emp_length.replace('< 1 year', 0.5)

x_test.emp_length = x_test.emp_length.replace('3 years', 3)

x_test.emp_length = x_test.emp_length.replace('1 year', 1)

x_test.emp_length = x_test.emp_length.replace('5 years', 5)

x_test.emp_length = x_test.emp_length.replace('4 years', 4)

x_test.emp_length = x_test.emp_length.replace('7 years', 7)

x_test.emp_length = x_test.emp_length.replace('8 years', 8)

x_test.emp_length = x_test.emp_length.replace('6 years', 6)

x_test.emp_length = x_test.emp_length.replace('9 years', 9)



x_train.emp_length.value_counts()
cats.remove('emp_length')

cats
x_train.home_ownership.value_counts()
x_train.home_ownership = x_train.home_ownership.replace('MORTGAGE', 443591)

x_train.home_ownership = x_train.home_ownership.replace('RENT', 356136)

x_train.home_ownership = x_train.home_ownership.replace('OWN', 87478)

x_train.home_ownership = x_train.home_ownership.replace('OTHER', 182)

x_train.home_ownership = x_train.home_ownership.replace('NONE', 50)

x_train.home_ownership = x_train.home_ownership.replace('ANY', 3)



x_test.home_ownership = x_test.home_ownership.replace('MORTGAGE', 443591)

x_test.home_ownership = x_test.home_ownership.replace('RENT', 356136)

x_test.home_ownership = x_test.home_ownership.replace('OWN', 87478)

x_test.home_ownership = x_test.home_ownership.replace('OTHER', 182)

x_test.home_ownership = x_test.home_ownership.replace('NONE', 50)

x_test.home_ownership = x_test.home_ownership.replace('ANY', 3)



x_train.home_ownership.value_counts()
cats.remove('home_ownership')

cats
oe = ce.OrdinalEncoder(cols=cats, return_df=False)



x_train[cats] = oe.fit_transform(x_train[cats])

x_test[cats] = oe.transform(x_test[cats])
x_train = x_train.drop('emp_title', axis=1)

x_test = x_test.drop('emp_title', axis=1)

x_train.head()
x_train.fillna(0,inplace=True)
x_train = x_train.loc[:,[

'loan_amnt',\

'installment',\

'grade',\

'sub_grade',\

'emp_length',\

'home_ownership',\

'annual_inc',\

'purpose',\

'title',\

'zip_code',\

'addr_state',\

'dti',\

'delinq_2yrs',\

'earliest_cr_line',\

'inq_last_6mths',\

'mths_since_last_delinq',\

'mths_since_last_record',\

'open_acc',\

'pub_rec',\

'revol_bal',\

'revol_util',\

'total_acc',\

'initial_list_status',\

'collections_12_mths_ex_med',\

'mths_since_last_major_derog',\

'application_type',\

'acc_now_delinq',\

'tot_coll_amt',\

'tot_cur_bal'

]]

x_test =   x_test.loc[:,[

'loan_amnt',\

'installment',\

'grade',\

'sub_grade',\

'emp_length',\

'home_ownership',\

'annual_inc',\

'purpose',\

'title',\

'zip_code',\

'addr_state',\

'dti',\

'delinq_2yrs',\

'earliest_cr_line',\

'inq_last_6mths',\

'mths_since_last_delinq',\

'mths_since_last_record',\

'open_acc',\

'pub_rec',\

'revol_bal',\

'revol_util',\

'total_acc',\

'initial_list_status',\

'collections_12_mths_ex_med',\

'mths_since_last_major_derog',\

'application_type',\

'acc_now_delinq',\

'tot_coll_amt',\

'tot_cur_bal'

]]

x_train.head()
scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(x_train, y_train))):

    x_train_, y_train_ = x_train.values[train_ix], y_train.values[train_ix]

    x_val, y_val = x_train.values[test_ix], y_train.values[test_ix]
import lightgbm as lgb

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

        importance_type='split', learning_rate=0.05, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

        random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

        subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



clf.fit(x_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(x_val, y_val)])

y_pred = clf.predict_proba(x_val)[:,1]

score = roc_auc_score(y_val, y_pred)

print(score)

fig, ax = plt.subplots(figsize=(10, 15))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain') # 変数重要をプロット


# 全データで再学習し、testに対して予測する

clf.fit(x_train, y_train)



y_pred = clf.predict_proba(x_test)[:,1] # predict_probaで確率を出力する





# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')