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

from scipy import stats

import math



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from lightgbm import LGBMClassifier,LGBMRegressor

import lightgbm as lgb

import gc

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



from sklearn.model_selection import train_test_split

#import featuretools as ft

#import featuretools.variable_types as vtypes
#ファイル読込

#df_train_ = pd.read_csv('../input/homework-for-students2/train.csv', index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])

#df_test = pd.read_csv('../input/homework-for-students2/test.csv',index_col=False,parse_dates=['issue_d', 'earliest_cr_line'])



#df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=False,parse_dates=['issue_d', 'earliest_cr_line'],skiprows=lambda x: x%20!=0)

df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=False)

#df_test = pd.read_csv('../input/exam-for-students20200129/test.csv',index_col=False,parse_dates=['issue_d', 'earliest_cr_line'],skiprows=lambda x: x%20!=0)

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv',index_col=False)



#追加特徴量の読み込み

#df_spi = pd.read_csv('../input/exam-for-students20200129/country_info.csv',parse_dates=['date'])

#df_spi = pd.read_csv('../input/exam-for-students20200129/country_info.csv')
##trainとテストにID列を付加⇒元のIndexを代用

df_train = df_train.reset_index()

df_train = df_train.rename(columns={'index':'ID'})

df_test = df_test.reset_index()

df_test = df_test.rename(columns={'index':'ID'})
#オブジェクトとそれ以外のデータを分けて持っておくと後々便利

date_feats = df_train.dtypes[df_train.dtypes == "datetime64[ns]"].index

numerical_feats = df_train.dtypes[(df_train.dtypes == 'float64') | (df_train.dtypes == 'int64')].index

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
#あとで見分けるため、Trainに0をTestに1を付加した列を追加する

df_train['Train_Test'] = 0

df_test['ConvertedSalary'] = np.nan

df_test['Train_Test'] = 1
#精度向上のためここでtrainとテストを一旦マージする

df_train = pd.concat([df_train,df_test],axis=0)

#インデックスを振りなおす

df_train = df_train.reset_index(drop=True)
#数も特徴量として使う

#df_train['missing_value_sum'] = df_train.isnull().sum(axis=1)

#df_test['missing_value_sum'] = df_test.isnull().sum(axis=1)
#objectデータも使いたいため、すべてＣｏｕｎｔエンコーディングをかける

for col in categorical_feats:

    #print(col)

    summary = df_train[col].value_counts()

    df_train[col] = df_train[col].map(summary)
#まずは正規化する

scaler_per = StandardScaler()

# 与えられた行列の各特徴量について､平均と標準偏差を算出

scaler_per.fit(df_train)

# Xを標準化した行列を生成

X_std = scaler_per.fit_transform(df_train.fillna(0))
# PCAのインスタンスを生成し、主成分を4つまで取得

pca = PCA(n_components=5) 

X_pca = pca.fit_transform(X_std)
df_pca = pd.DataFrame(data=X_pca, index=df_train.index, dtype='float64',columns=['pca_per_1','pca_per_2','pca_per_3','pca_per_4','pca_per_5'])
# PCAの結果を結合

df_train = pd.concat([df_train,df_pca],axis=1)
# testデータを再抽出してIDで元の順番にソートする

X_test = df_train[df_train['Train_Test']==1].copy()

X_test = X_test.sort_values('ID', ascending=True)

X_test = X_test.set_index('ID')

y_test = X_test['ConvertedSalary'].copy()

X_test.drop(['ConvertedSalary'],inplace=True,axis=1)
#学習データ取得

X_train = df_train[df_train['Train_Test']==0].copy()

X_train = X_train.set_index('ID')

#教師ラベルを取得

y_train = X_train['ConvertedSalary'].copy()

#不要な列削除

X_train.drop(['ConvertedSalary'],inplace=True,axis=1)
#df_train全体で学習するので修正

#df_y_train = df_train['ConvertedSalary'].copy()

#df_x_train = df_train.drop(columns='ConvertedSalary',axis=1).copy()
#今回のメトリックスがＲＭＳＬＥなので目的変数を対数変換する

y_train = np.log1p(y_train)
# 訓練データとテストデータに分割する

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train)



# データセットを生成する

lgb_train = lgb.Dataset(X_train_, y_train_)

lgb_eval = lgb.Dataset(X_test_, y_test_, reference=lgb_train)



# LightGBM のハイパーパラメータ

lgbm_params = {

    # 回帰問題

    'objective': 'regression',

    # RMSE (平均二乗誤差平方根) の最小化を目指す

    'metric': 'rmse',

    'random_state': '71',

    'class_weight': 'None', 

    'colsample_bytree': '0.05',

    'importance_type': 'split', 

    'learning_rate': '0.05', 

    'max_depth': '-1',

    'min_child_samples': '20', 

    'min_child_weight': '0.001', 

    'min_split_gain': '0.0',    

    'n_jobs': '-1', 

    'num_leaves': '15',

    'reg_alpha': '1.0', 

    'reg_lambda': '20.0', 

    'silent': 'True',

    'subsample': '1.0', 

    'subsample_for_bin': '200000', 

    'subsample_freq': '0',

}



# 上記のパラメータでモデルを学習する

model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval,categorical_feature = categorical_feats.values.tolist())



# テストデータを予測する

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

y_pred_result = np.expm1(y_pred)
submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred_result

submission.to_csv('submission.csv')