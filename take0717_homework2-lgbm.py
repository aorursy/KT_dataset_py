###

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm
df_train = pd.read_csv('../input/train.csv',index_col = 0)

#df_train = pd.read_csv('../input/train_small.csv',index_col = 0)

df_test = pd.read_csv('../input/test.csv',index_col = 0)

df_gdp = pd.read_csv('../input/US_GDP_by_State.csv')

df_sll = pd.read_csv('../input/statelatlong.csv')
df_train['cr_line_m'] = df_train['earliest_cr_line'].str[:3]

df_test['cr_line_m'] = df_test['earliest_cr_line'].str[:3]



df_train['cr_line_y'] = df_train['earliest_cr_line'].str[-4:]

df_test['cr_line_y'] = df_test['earliest_cr_line'].str[-4:]
df_train.loc[df_train['title'].astype('str').str.contains('Debt consolidation'), 'title_flg'] = '1'

df_train.loc[df_train['title'].astype('str').str.contains('Credit card refinancing'), 'title_flg'] = '1'



df_test.loc[df_test['title'].astype('str').str.contains('Debt consolidation'), 'title_flg'] = '1'

df_test.loc[df_test['title'].astype('str').str.contains('Credit card refinancing'), 'title_flg'] = '1'
df_gdp = df_gdp.drop("year",axis=1)

df_gdp_mean = df_gdp.rename(columns={'State':'City'})

df_gdp_mean = df_gdp_mean.groupby(['City']).mean()

df_gdp_mean = df_gdp_mean.reset_index()
df_sll = df_sll.rename(columns={'State':'addr_state'})
df_train = pd.merge(df_train, df_sll, how='left')

df_train = pd.merge(df_train, df_gdp_mean, how='left')



df_test = pd.merge(df_test, df_sll, how='left')

df_test = pd.merge(df_test, df_gdp_mean, how='left')
y_train = df_train.loan_condition

x_train = df_train.drop(['loan_condition','issue_d','application_type','acc_now_delinq'],axis=1)



x_test = df_test.drop(['issue_d','application_type','acc_now_delinq'],axis=1)
cats = []

nums = []



for col in x_train.columns:

    if x_train[col].dtype == 'object':

        cats.append(col)

    else:

        nums.append(col)
#x_train.annual_inc = x_train.annual_inc.fillna(x_test.annual_inc.min())

x_train[nums] = x_train[nums].fillna(x_train[nums].min())

x_train[cats] = x_train[cats].fillna("NaN")



#x_test.annual_inc = x_test.annual_inc.fillna(x_test.annual_inc.min())

x_test[nums] = x_test[nums].fillna(x_test[nums].min())

x_test[cats] = x_test[cats].fillna("NaN")
oe = ce.OrdinalEncoder(cols = cats,return_df=False)



x_train[cats] = oe.fit_transform(x_train[cats])

x_test[cats] = oe.transform(x_test[cats])
x_train.annual_inc = x_train.annual_inc.apply(np.log1p)

x_test.annual_inc = x_test.annual_inc.apply(np.log1p)
scaler = StandardScaler()

scaler.fit(x_train[nums])

x_train[nums] = scaler.transform(x_train[nums])

x_test[nums] = scaler.transform(x_test[nums])
#トレーニングデータとバリデーションデータに分割

#x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=71)



skf = StratifiedKFold(n_splits=15, random_state=71, shuffle=True)



X=x_train

y=y_train



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X, y))):

    x_train_, y_train_ = X.iloc[train_ix], y.iloc[train_ix]

    x_val, y_val = X.iloc[test_ix], y.iloc[test_ix]



    

#トレーニングデータとバリデーションデータに分割

#from sklearn.model_selection import KFold

#kf = KFold(n_splits=5, random_state=71, shuffle=True)

#for i,(train_ix, test_ix) in enumerate(tqdm(kf.split(X, y))):

#    x_train_, y_train_ = X.iloc[train_ix], y.iloc[train_ix]

#    x_val, y_val = X.iloc[test_ix], y.iloc[test_ix]
import lightgbm as lgb

import matplotlib.pyplot as plt

# 数値変数を順位ないし相対順位に変換

params = {

'boosting_type': 'gbdt',

# other params here

}

train_set = lgb.Dataset(x_train, y_train)

val_set = lgb.Dataset(x_val, y_val)

# 学習

model = lgb.train(params, train_set, 500, val_set, verbose_eval=100)

# 可視化

fig, ax = plt.subplots(figsize=(10, 15))

lgb.plot_importance(model, max_num_features=50, ax=ax, importance_type='gain') # 'gaiのn'他に'split'がある。

imp = model.feature_importance(importance_type='gain') # importancをenumpy arrayで受け取る

#use_col = x_train.columns[imp > th] # 閾値を切るなり、sortして必要な数に絞り込むなりする。
#LGBM（パラメータ指定有り）

import lightgbm as lgb

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

  importance_type='split', learning_rate=0.02, max_depth=-1,

  min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

  n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

  random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

  subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



clf.fit(x_train_, y_train_, early_stopping_rounds=200,eval_metric='auc', eval_set=[(x_val, y_val)])



y_pred = clf.predict_proba(x_val)[:,1]

score = roc_auc_score(y_val, y_pred)

print(score)
# 全データで再学習し、testに対して予測する

#clf.fit(x_train, y_train, early_stopping_rounds=300, eval_metric='auc', eval_set=[(x_val, y_val)])



y_pred = clf.predict_proba(x_test)[:,1] # predict_probaで確率を出力する
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')