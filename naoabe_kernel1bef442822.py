# パッケージ読み込み



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import KFold



from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

from lightgbm import LGBMRegressor



from hyperopt import fmin, tpe, hp, rand, Trials



import gc
# 表示列数上限を500列に変更

pd.set_option('display.max_columns', 500)
# データセットの読み込み

#df_train = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, skiprows=lambda x: x%20!=0)

#df_train = pd.read_csv('./multiseries3_train.csv',parse_dates=['日付'])

df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=0)

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv', index_col=0)
# 行・列数をカウント

df_train.shape, df_test.shape
# 中身を確認(train)

df_train.head()
# 中身を確認(test)

df_test.head()
# 基礎統計量を確認(train)

df_train.describe()
# 基礎統計量を確認(test)

df_test.describe()
# 欠損値をカウント(train)

df_train.isnull().sum()
# 欠損値をカウント(test)

df_test.isnull().sum()
# 特徴量のヒストグラムを確認

f = 'ConvertedSalary'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=1, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
# 特徴量の項目別の件数と構成比を確認（train全件）

for col_name in df_train.columns:

    print(df_train[col_name].value_counts())

    print(df_train[col_name].value_counts() / len(df_train[col_name]))
# 特徴量の項目別の件数と構成比を確認（test全件）

for col_name in df_test.columns:

    print(df_test[col_name].value_counts())

    print(df_test[col_name].value_counts() / len(df_test[col_name]))
# Countryデータ読み込み

df_country_info = pd.read_csv('../input/exam-for-students20200129/country_info.csv')
df_country_info
df_country_info['Country'].value_counts()
df_country_info_edit = df_country_info[['Country', 'Region']].copy()

df_country_info_edit
# 学習データ＆テストデータにCountry情報を追加

df_train_add_countryinfo = pd.merge(df_train, df_country_info_edit, how = 'left', on = ['Country']).copy()

df_test_add_countryinfo = pd.merge(df_test, df_country_info_edit, how = 'left', on = ['Country']).copy()
df_train_add_countryinfo
df_test_add_countryinfo
#Xとyに分割



#y_train = np.log1p(df_train['IncurredClaims'])



# Counryを除く

y_train = df_train_add_countryinfo['ConvertedSalary'].copy()

X_train = df_train_add_countryinfo.drop(['ConvertedSalary', 'Country'], axis=1).copy()

X_test = df_test_add_countryinfo.drop(['Country'], axis=1).copy()
y_train
X_train.head()
X_test
# dtypeがobjectのカラム名とユニーク数を確認

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)        

        print(col, X_train[col].nunique())
# dtypeがobjectのカラムを全てOrdinalエンコーディング

encoder = OrdinalEncoder(cols=cats)

encoder
X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])
X_train
y_train.head()
X_test
X_test['Currency']
#ターゲットエンコーディング



target = 'ConvertedSalary'

t_encoding_col = ['Region', 'Employment', 'LastNewJob', 'YearsCodingProf', 'SalaryType', 'Currency', 'Age', 'Student', 'CompanySize', 'MilitaryUS', 'CareerSatisfaction', 'NumberMonitors', 'OperatingSystem', 'EducationParents']



for i, t_col in enumerate(t_encoding_col):



    X_temp = pd.concat([X_train, y_train], axis=1)



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([t_col])[target].mean()

    enc_test = X_test[t_col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([t_col])[target].mean()

        enc_train.iloc[val_ix] = X_val[t_col].map(summary)

        

    X_train[t_col] = enc_train

    X_test[t_col] = enc_test
X_train
X_test
# ターゲットをRMSLE用に対数変換

y_train_log = np.log1p(y_train).copy()
y_train_log
# 層化抽出でモデリング



scores = []



#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

kf = KFold(n_splits=5, random_state=71, shuffle=True)



#for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_train_log))):

    X_train_, y_train_ = X_train.values[train_ix], y_train_log.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train_log.values[test_ix]

    

    reg = LGBMRegressor()

    

#    reg.fit(X_train_, np.log1p(y_train_), eval_metric='rmse')

    reg.fit(X_train_, y_train_, eval_metric='rmse')

    y_pred = reg.predict(X_val)

#    y_pred = np.expm1(reg.predict(X_val))

    

    score = mean_absolute_error(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))

    

print(np.mean(scores))

print(scores)
np.expm1(y_pred)
# 全データで再学習し、testに対して予測する

reg.fit(X_train, y_train_log, eval_metric='rmse')

#reg.fit(X_train, y_train, eval_metric='rmse')

y_pred = np.expm1(reg.predict(X_test))

#y_pred = reg.predict(X_test)
y_pred
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

#submission = pd.read_csv('../input/sample/sample_submission.csv', index_col=0)

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')
# Feature Importance

fti = reg.feature_importances_   



print('Feature Importances:')

for i, feat in enumerate(X_train):

    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(reg, max_num_features=30, ax=ax, importance_type='gain')