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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from scipy import stats

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso

)

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

# データの読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ

# 学習データとテストデータのマージ

train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

alldata[na_col_list].dtypes.sort_values() #データ型
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
alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
# カテゴリカル変数の特徴量をリスト化

cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()

# 数値変数の特徴量をリスト化

num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()

# データ分割および提出時に必要なカラムをリスト化

other_cols = ['Id','WhatIsData']

# 余計な要素をリストから削除

cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去

num_cols.remove('Id') #Id削除

# カテゴリカル変数をダミー化

alldata_cat = pd.get_dummies(alldata[cat_cols])

# データ統合

all_data = pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)
# マージデータを学習データとテストデータに分割

train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

# 学習データ内の分割

train_x = train_.drop('SalePrice',axis=1)

train_y = np.log(train_['SalePrice'])

# テストデータ内の分割

test_id = test_['Id']

test_data = test_.drop('Id',axis=1)
scaler = StandardScaler()  #スケーリング

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    ls = Lasso(alpha=alpha) #Lasso回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
plt.subplots_adjust(wspace=0.4)

plt.subplot(121)

plt.scatter(np.exp(y_train),np.exp(best_estimator.predict(X_train)))

plt.subplot(122)

plt.scatter(np.exp(y_test),np.exp(best_estimator.predict(X_test)))


ls = Lasso(alpha = 0.01)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_x,train_y)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_data)),columns=['Saleprice'])

test_Id = pd.DataFrame(test_id,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('submission.csv',index=False)