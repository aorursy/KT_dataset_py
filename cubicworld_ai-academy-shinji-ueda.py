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
from matplotlib import pyplot as plt

%matplotlib inline



from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

df_train = pd.read_csv('/kaggle/input/exam-for-students20200527/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/exam-for-students20200527/test.csv', index_col=0)
# データの確認(train)



display(df_train.shape)

display(df_train.dtypes)

display(df_train.index)

display(df_train.head())
# データの確認(test)



display(df_test.shape)

display(df_test.dtypes)

display(df_test.index)

display(df_test.head())
# 基礎統計量（train）



display(df_train.describe())

display(df_train.describe(exclude='number'))

display(df_train.isnull().sum())
# 基礎統計量（test）



display(df_test.describe())

display(df_test.describe(exclude='number'))

display(df_test.isnull().sum())
# TradePriceの分布の確認



target = 'TradePrice'



plt.figure(figsize=(15,10))

df_train[target].hist(density=True, alpha=0.5, bins=10)

plt.xlabel(target)

plt.ylabel('density')

plt.show()
# 数値系カラムに対して分布を確認



num_cols = []



for col in df_train.columns:

    if df_train[col].dtype != 'object':

        num_cols.append(col)



display(num_cols)
# id,FrontageIsGreaterFlag以外の数値カラムの値をヒストグラムで表示



num_cols.remove('FrontageIsGreaterFlag')

num_cols.remove('TradePrice')



display(len(num_cols))



for col in num_cols:

    plt.figure(figsize=(5,5))

    df_train[col].hist(density=True, alpha=0.5)

    df_test[col].hist(density=True, alpha=0.5)

    plt.xlabel(col)

    plt.ylabel('density')

    plt.show()



    

# trainにおける各カラムとターゲットとの相関関係を確認



for col in num_cols:

    df_train.plot.scatter(x=col, y=target, figsize=(3,3))
# XとYとに分割



y_train = df_train.TradePrice

X_train = df_train



X_test = df_test
display(y_train.head())

display(y_train.shape)
display(X_train.shape)

display(X_train.head())
display(X_train.describe())
cat = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cat.append(col)

        print(col, X_train[col].nunique())
for col in cat:

    print(col, X_test[col].nunique())
# サブテーブルの読み込み



df_city = pd.read_csv('/kaggle/input/exam-for-students20200527/city_info.csv')



print(df_city.head())
df_station = pd.read_csv('/kaggle/input/exam-for-students20200527/station_info.csv')



print(df_station)
# サブテーブルの結合 city

display(X_train.shape)

display(X_test.shape)



X_train.drop('Prefecture', axis=1, inplace=True)

X_test.drop('Prefecture', axis=1, inplace=True)

df_city.drop('Prefecture', axis=1, inplace=True)

X_train = pd.merge(X_train, df_city, left_on='Municipality', right_on='Municipality', how='left')

X_test = pd.merge(X_test, df_city, left_on='Municipality', right_on='Municipality', how='left')



X_train = X_train.rename(columns={'Latitude':'c_Latitude', 'Longitude':'c_Longitude'})

X_test = X_test.rename(columns={'Latitude':'c_Latitude', 'Longitude':'c_Longitude'})



X_train.drop('Municipality', axis=1, inplace=True)

X_test.drop('Municipality', axis=1, inplace=True)



display(X_train.shape)

display(X_train.head())

display(X_test.shape)

display(X_test.head())
# サブテーブルの結合 station

display(X_train.shape)

display(X_test.shape)



X_train = pd.merge(X_train, df_station, left_on='NearestStation', right_on='Station', how='left')

X_test = pd.merge(X_test, df_station, left_on='NearestStation', right_on='Station', how='left')



X_train = X_train.rename(columns={'Latitude':'s_Latitude', 'Longitude':'s_Longitude'})

X_test = X_test.rename(columns={'Latitude':'s_Latitude', 'Longitude':'s_Longitude'})



X_train.drop('NearestStation', axis=1, inplace=True)

X_test.drop('NearestStation', axis=1, inplace=True)

X_train.drop('Station', axis=1, inplace=True)

X_test.drop('Station', axis=1, inplace=True)



display(X_train.shape)

display(X_train.head())

display(X_test.shape)

display(X_test.head())
# 不要な列の削除



X_train.drop('DistrictName', axis=1, inplace=True)

X_test.drop('DistrictName', axis=1, inplace=True)



display(X_train.head())
# object型の値を再確認



display(X_train.describe(exclude='number'))

display(X_test.describe(exclude='number'))
display(X_train['Region'].value_counts())

display(X_test['Region'].value_counts())
# Regionをカウントエンコーディング



summary = X_train['Region'].value_counts()



X_train['Region'] = X_train['Region'].map(summary)

X_test['Region'] = X_test['Region'].map(summary)



display(X_train.head())
# FloorPlanを確認



display( X_train['FloorPlan'].value_counts(), X_train['FloorPlan'].nunique() )

display( X_test['FloorPlan'].value_counts(), X_test['FloorPlan'].nunique()  )

# FloorPlan を辞書に基づいてラベルエンコーディング



mappingdict = {

    "FloorPlan": {

        "1R": 11,

        "1R+S": 12,

        "1K": 13,

        "1K+S": 14,

        "1DK":15,

        "1DK+S":16,

        "1LDK":17,

        "1LDK+S":18,

        "2K":21,

        "2K+S":22,

        "2LK":23,

        "2LK+S":24,

        "2DK":25,

        "2DK+S":26,

        "2LD":27,

        "2LD+S":28,

        "2LDK":29,

        "2LDK+S":30,

        "3K":31,

        "3LK":32,

        "3LD":33,

        "3DK":34,

        "3DK+S":35,

        "3LDK":36,

        "3LDK+K":37,

        "3LDK+S":38,

        "4K":41,

        "4DK":42,

        "4DK+S":43,

        "4LDK":44,

        "4LDK+S":45,

        "4LDK+K":46,

        "5DK":51,

        "5LDK":52,

        "5LDK+S":53,

        "5LDK+K":54,

        "6LDK":61,

        "Open Floor":81,

        "Duplex":91,

        "Studio Apartment":99

    }

}



X_train = X_train.replace(mappingdict)

X_test = X_test.replace(mappingdict)



X_train['FloorPlan'] = X_train['FloorPlan'].astype(float)

X_test['FloorPlan'] = X_test['FloorPlan'].astype(float)



display(X_train.head())
# LandShapeを確認



display( X_train['LandShape'].value_counts() )

display( X_test['LandShape'].value_counts() )
# LandSpane をカウントエンコーディング



summary = X_train['LandShape'].value_counts()



X_train['LandShape'] = X_train['LandShape'].map(summary)

X_test['LandShape'] = X_test['LandShape'].map(summary)



display(X_train.head())
# Structureを確認



display( X_train['Structure'].value_counts() )

display( X_test['Structure'].value_counts() )
# Structureはラベルエンコーディング



encoder = OrdinalEncoder(return_df=False)



X_train['Structure'] = encoder.fit_transform(X_train['Structure'].values)

X_test['Structure'] = encoder.transform(X_test['Structure'].values)



display(X_train.head())
# Use, Purpose



display(X_train['Use'].value_counts())

display(X_test['Use'].value_counts())

display(X_train['Purpose'].value_counts())

display(X_test['Purpose'].value_counts())

# Use, Purposeはラベルエンコーディング



encoder = OrdinalEncoder(return_df=False)



X_train['Use'] = encoder.fit_transform(X_train['Use'].values)

X_test['Use'] = encoder.transform(X_test['Use'].values)



X_train['Purpose'] = encoder.fit_transform(X_train['Purpose'].values)

X_test['Purpose'] = encoder.transform(X_test['Purpose'].values)



display(X_train.head())
# object型を確認



display(X_train.describe(exclude='number'))
# Type



display(X_train['Type'].value_counts())

display(X_test['Type'].value_counts())

# Typeをカウントエンコーディング



summary = X_train['Type'].value_counts()



X_train['Type'] = X_train['Type'].map(summary)

X_test['Type'] = X_test['Type'].map(summary)



display(X_train.head())

display(X_train.describe(exclude='number'))
# Directionを確認



display(X_train['Direction'].value_counts())

display(X_test['Direction'].value_counts())
# Direction を辞書に基づいてラベルエンコーディング



mappingdict = {

    "Direction": {

        "South": 11,

        "Southeast": 12,

        "Southwest":13,

        "East":21,

        "West":31,

        "North":41,

        "Northeast":42,

        "Northwest":43,

        "No facing road":51

    }

}



X_train = X_train.replace(mappingdict)

X_test = X_test.replace(mappingdict)



X_train['Direction'] = X_train['Direction'].astype(float)

X_test['Direction'] = X_test['Direction'].astype(float)



display(X_train.head())

display(X_train.describe(exclude='number'))
# Classification



display(X_train['Classification'].value_counts())

display(X_test['Classification'].value_counts())
# Classificationはラベルエンコーディング



encoder = OrdinalEncoder(return_df=False)



col = 'Classification'



X_train[col] = encoder.fit_transform(X_train[col].values)

X_test[col] = encoder.transform(X_test[col].values)



display(X_train.head())

display(X_train.describe(exclude='number'))
# CityPlanning



display(X_train['CityPlanning'].value_counts())

display(X_test['CityPlanning'].value_counts())
# CityPlanningをカウントエンコーディング



col = 'CityPlanning'



summary = X_train[col].value_counts()



X_train[col] = X_train[col].map(summary)

X_test[col] = X_test[col].map(summary)



display(X_train.head())

display(X_train.describe(exclude='number'))
# Renovation



display(X_train['Renovation'].value_counts())

display(X_test['Renovation'].value_counts())
# Renovation を辞書に基づいてラベルエンコーディング



col = 'Renovation'



mappingdict = {

    "Renovation": {

        "Done": 1,

        "Not yet": 0

    }

}



X_train = X_train.replace(mappingdict)

X_test = X_test.replace(mappingdict)



X_train[col] = X_train[col].astype(float)

X_test[col] = X_test[col].astype(float)



display(X_train.head())

display(X_train.describe(exclude='number'))
# Remarks



display(X_train['Remarks'].value_counts())

display(X_test['Remarks'].value_counts())
# Remarksをカウントエンコーディング



col = 'Remarks'



summary = X_train[col].value_counts()



X_train[col] = X_train[col].map(summary)

X_test[col] = X_test[col].map(summary)



display(X_train.head())

display(X_train.describe(exclude='number'))
# TimeToNearestStation



display(X_train['TimeToNearestStation'].value_counts())

display(X_test['TimeToNearestStation'].value_counts())
# TimeToNearestStationはdrop



X_train.drop('TimeToNearestStation', axis=1, inplace=True)

X_test.drop('TimeToNearestStation', axis=1, inplace=True)



display(X_train.head())

display(X_train.describe(exclude='number'))
# FrontageIsGreaterFlag



display(X_train['FrontageIsGreaterFlag'].value_counts())

display(X_test['FrontageIsGreaterFlag'].value_counts())
X_train['FrontageIsGreaterFlag'] = X_train['FrontageIsGreaterFlag'].astype(int)

X_test['FrontageIsGreaterFlag'] = X_test['FrontageIsGreaterFlag'].astype(int)
display(X_train.head())

#display(X_train.describe(exclude='number'))
display(X_train.dtypes)

display(X_test.dtypes)
# 欠損値処理



X_train.isnull().sum()
X_train.fillna(-9999, inplace=True)

X_train.fillna(-9999, inplace=True)



X_train.isnull().sum()
import lightgbm as lgbm

from sklearn.model_selection import train_test_split
# Validation



train_set, val_set = train_test_split(X_train, test_size=0.2, random_state=71)
train_set.shape
train_set.head()
l_x_train = train_set.drop('TradePrice', axis=1)

l_y_train = train_set['TradePrice']



l_x_val = val_set.drop('TradePrice', axis=1)

l_y_val = val_set['TradePrice']

lgb_train = lgbm.Dataset(l_x_train, l_y_train)

lgb_eval = lgbm.Dataset(l_x_val, l_y_val)
params = {'metric': 'rmse', 'max_depth' : 9}
gbm = lgbm.train(params,

                lgb_train,

                valid_sets=lgb_eval,

                num_boost_round=10000,

                early_stopping_rounds=100,

                verbose_eval=50)
y_pred = gbm.predict(X_test)

# submission



submission = pd.read_csv('/kaggle/input/exam-for-students20200527/sample_submission.csv')
submission.head()
submission['TradePrice'] = y_pred
submission['TradePrice'] = submission['TradePrice'].round(0)



submission.head()


submission.to_csv('./submission.csv', header=True, index=False)