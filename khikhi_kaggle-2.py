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
train_df = pd.read_csv('../input/exam-for-students20200923/train.csv')
test_df = pd.read_csv('../input/exam-for-students20200923/test.csv')
country_info_df = pd.read_csv('../input/exam-for-students20200923/country_info.csv')
country_info_df.dtypes
country_info_df.head()
# countryの数字にカンマが入っているので.に変えてNum化-1
count_cols = country_info_df.columns
count_cols = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service']
# countryの数字にカンマが入っているので.に変えてNum化-2
for ccol in count_cols:
    country_info_df[ccol] = country_info_df[ccol].apply(lambda x: float(x.replace(",", ".")) if type(x) is str else float(x))

country_info_df.head()
country_info_df.dtypes
print(train_df.shape)
print(test_df.shape)
train_df['JobContactPriorities2'].dtypes
# country情報をマージ
train_df_new = pd.merge(train_df, country_info_df, left_on='Country', right_on='Country')
test_df_new = pd.merge(test_df, country_info_df, left_on='Country', right_on='Country')
print(train_df_new.shape)
print(test_df_new.shape)
train_df_new.dtypes
train_df_new.head()
test_df_new.head()
import matplotlib.pyplot as plt
import seaborn as sns
# ターゲット変数の分布チェック
# figsizeでグラフのサイズ調整
plt.figure(figsize=(20,10))
sns.distplot(train_df_new['ConvertedSalary'])
#見たところかなり高いサラリーがいる。モデルを分割することも検討
#　ターゲット指定
target = 'ConvertedSalary'

# トレーニングデータの作成
y_train = train_df_new[target]
X_train = train_df_new.drop(target, axis=1)


# テストデータの作成
X_test = test_df
print(X_train.shape)
print(X_test.shape)
xshape = X_train.shape[0]

print(xshape)
# カテゴリ化するときにTrainデータとTestデータに入っているカテゴリがずれると、ダミー変数の数が変わってしまうので、一旦Concat
concat_X = pd.concat([X_train, X_test], axis=0)

concat_X.shape
# Object型のカラムがいくつユニークな値を持っているかを確認

# objectカラム抜き出し
obj_cols = concat_X.select_dtypes(include = 'object')

# カラムごとのユニークな値数表示(30未満)
for objcol in obj_cols:
    if concat_X[objcol].nunique() < 30:
        print(objcol + ' : ' + str(concat_X[objcol].nunique()))
# カラムごとのユニークな値数表示（30以上）

for objcol in obj_cols:
    if concat_X[objcol].nunique() >= 30:
        print(objcol + ' : ' + str(concat_X[objcol].nunique()))
# Countryはマージしたので削除
concat_X.drop(['Country','Respondent'], axis=1, inplace=True)

# ３０以上はとりあえずDrop
concat_X.drop(['DevType','CurrencySymbol','FrameworkWorkedWith','RaceEthnicity'], axis=1, inplace=True)
#concat_X.drop(['DevType','CurrencySymbol','CommunicationTools','FrameworkWorkedWith','RaceEthnicity'], axis=1, inplace=True)
# CommunicationTools　はカンマ区切りで入っている。カンマでセパレートしてユニークにする
CommunicationTools_uniques = concat_X['CommunicationTools'].unique()

# 区切るよ。　ComToolsがユニークな値
ComTools = []


for comtooluniq in CommunicationTools_uniques:
#    print(CommunicationToolsunique, type(CommunicationToolsunique))
    if isinstance(comtooluniq, str):
        for word in comtooluniq.split(';'):
            ComTools.append(str.strip(word))
      
ComTools = np.unique(ComTools)
#print('ComTools : ' + ComTools)
print(ComTools)
# カラムを作って値を入れる
for i, comtool in enumerate(ComTools):
    comcol_name = 'commu_'+str(i)
#for comtool in ComTools:
    concat_X[comcol_name] = concat_X['CommunicationTools'].str.contains(comtool)
    
# CommunicationToolsはもういらない
concat_X.drop('CommunicationTools', axis=1, inplace=True)
#カラムが減っているかを確認
concat_X.shape
# 改めてカテゴリ数が３０個未満のカラムを拾って、カテゴリ型に変換

cat_cols = []

for col in concat_X.columns:
    if concat_X[col].nunique() < 30:
        cat_cols.append(col)

print(cat_cols)
concat_X[cat_cols] = concat_X[cat_cols].astype('category')
concat_X.dtypes
# トレーニングとテストに再分割
X_train = concat_X[:xshape]
X_test = concat_X[xshape:]
print(X_train.shape)
print(X_test.shape)
X_train.head()
# RMSLE対策。学習前にy_trainに、log(y+1)で変換することで、RMSEそのまま使う
#y_train = np.log(y_train + 1) 
# LightGBMパラメータ(CV)
# optunaによるハイパーパラメータ最適化用に、パラメータ設定を最低限に変更
from sklearn.model_selection import KFold

import optuna
import optuna.integration.lightgbm as lgb
#from sklearn.metrics import log_loss

# optunaのログを止める（プログレスバーがばらばらになるので）
optuna.logging.set_verbosity(optuna.logging.WARNING)

# rmse使うならmseしかないので、ラストにsqrt(ルート)してください
from sklearn.metrics import mean_squared_error
from math import sqrt




params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
}


dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols, free_raw_data=False)

tuner = lgb.LightGBMTunerCV(
        params, dtrain, verbose_eval=100, early_stopping_rounds=5, folds=KFold(n_splits=5)
    
#一時的に学習期間の短縮を図ります
#        params, dtrain, verbose_eval=100, early_stopping_rounds=2, folds=KFold(n_splits=2)
)

tuner.run()

#　上記を回した結果。BestParams
#best_params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'feature_pre_filter': False, 'lambda_l1': 0.2589659402934777, 'lambda_l2': 0.08181674871353375, 'num_leaves': 31, 'feature_fraction': 0.8999999999999999, 'bagging_fraction': 0.8268629732673147, 'bagging_freq': 2, 'min_child_samples': 20}
#  モデルの最適化されたパラメータ確認
print("Best score:", tuner.best_score)
best_params = tuner.best_params
print("Best params:", best_params)
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

X_train.head()
cat_cols
import lightgbm as lgb2

#全データ利用するのでダミーのValidデータを用意train,val　に分けます
from sklearn.model_selection import train_test_split

X_train_dummy, X_valid_dummy, y_train_dummy, y_valid_dummy = train_test_split(X_train, y_train, test_size=0.2)

evaluation_results = {}

lgb_train = lgb2.Dataset(X_train, y_train) 
lgb_eval = lgb2.Dataset(X_valid_dummy, y_valid_dummy, reference=lgb_train)

model = lgb2.train(best_params, lgb_train, valid_sets=[lgb_train, lgb_eval], valid_names=['train', 'valid'], evals_result=evaluation_results,
verbose_eval=-1, num_boost_round=1000, early_stopping_rounds=1000)
# 投稿用CSV作成
submit_df = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv' , index_col=0)
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred
#RMSLE対策。逆関数でもとに戻す
#y_pred = np.exp(y_pred) - 1 
y_pred
submit_df[target] = y_pred
#submit_df[target] = df_cv_avg['rslt_avg2']
submit_df.to_csv('./submission11.csv')
submit_df.head()

