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
# 必要そうなライブラリ読み込み

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
%matplotlib inline
# データの読み込み
# 訓練データ
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
# テストデータ
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv') 

# 一括で欠損値保管するため一旦マージ
# ただし、あとで分割するためWhatIsDataにタグ残し
train_data['WhatIsData'] = 'Train'
test_data['WhatIsData'] = 'Test'

# テストデータにSalePriceは存在しなため、ダミーデータ挿入
test_data['SalePrice'] = 9999999999

# マージ処理
alldata = pd.concat([train_data,test_data],axis=0).reset_index(drop=True)

print('The size of train_data is : ' + str(train_data.shape))
print('The size of test_data is : ' + str(test_data.shape))
print('The size of alldata is : ' + str(alldata.shape))
# 欠損を含むカラムをリスト化
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist()
# データ型に応じて欠損値を補完する

# floatの場合は0
# objectの場合は'NA'
na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist() #float64
na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist() #object

# float64型で欠損している場合は0を代入
for na_float_col in na_float_cols:
    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

# object型で欠損している場合は'NA'を代入
for na_obj_col in na_obj_cols:
    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'
# カテゴリカル変数の特徴量をリスト化
cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()
# 数値変数の特徴量をリスト化
num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()
# データ分割および提出時に必要なカラムをリスト化
other_cols = ['Id','WhatIsData']
# 余計な要素をリストから削除
cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去
num_cols.remove('Id') #Id削除

# カテゴリ変数に対してLabel Encodingの実施
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

encoded_data = alldata[cat_cols].apply(le.fit_transform)

# データ統合
all_data = pd.concat([alldata[other_cols],alldata[num_cols],encoded_data],axis=1)
print(all_data)
# マージデータを学習データとテストデータに再分割
train_ = all_data[alldata['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
test_ = all_data[alldata['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)


# 学習データ内の分割
train_x = train_.drop('SalePrice',axis=1)
train_y_ = train_['SalePrice']
train_y = np.log(train_['SalePrice'])

# テストデータ内の分割
test_id = test_['Id']
test_x= test_.drop('Id',axis=1)
# データの分割
train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.3,
        random_state=0,
        )
import lightgbm as lgb

lgb_train = lgb.Dataset(train_x, train_y,
                       categorical_feature=cat_cols)
lgb_eval = lgb.Dataset(valid_x, valid_y,
                       reference=lgb_train,
                       categorical_feature=cat_cols)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 128,
    'learning_rate': 0.01,
    'feature_fraction': 0.38,
    'bagging_fraction': 0.68,
    'bagging_freq': 5,
    'verbose': 0
}

model = lgb.train(params,
                  lgb_train,
                  valid_sets=[lgb_train, lgb_eval],
                 verbose_eval=10,
                 num_boost_round=5000,
                 early_stopping_rounds=1000,)

y_pred = model.predict(test_x, 
                      num_iteration=model.best_iteration)
# 結果の出力

my_submission = pd.DataFrame()
my_submission["Id"] = test_id
my_submission["SalePrice"] = np.exp(model.predict(test_x))
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
