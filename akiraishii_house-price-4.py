# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats

from scipy.stats import norm, skew



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root_dir = '/kaggle/input/house-prices-advanced-regression-techniques/'

print(os.listdir('/kaggle/input/house-prices-advanced-regression-techniques/'))
'''for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames :'''
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train_df.head()
train_df.info()
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
submit_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submit_df.head()
test_df.info()
sns.distplot(train_df['SalePrice'], fit=norm)
mu, sigma = norm.fit(train_df['SalePrice'])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot = plt)
#Category columnsの列を作成する。

#これらはcountplotで描画する

category_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 

                   'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 

                   'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 

                   'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

                   'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 

                   'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 

                   'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 

                   'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

                   'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
#

for col in train_df.columns[1:-1]:

    fig = plt.figure()

    #print(col)

    if col in category_columns:

        sns.boxplot(col, y='SalePrice', data=train_df)

    else:

        sns.scatterplot(col, y='SalePrice', data=train_df)
#歪度が右になっているものを正規分布の形に近づけるための変換

#log(x+1)をすることで変換する

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])



sns.distplot(train_df['SalePrice'], fit=norm)



mu, sigma = norm.fit(train_df['SalePrice'])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], 

          loc='bext')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)
#数値とみなしても良さそうなカテゴリカラムの一覧

num_category_columns = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LandSlope', 

                        'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 

                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 

                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 

                        'GarageCond', 'PavedDrive', 'PoolQC']
train_df.head()
train_df.columns[-1]
#Create Numeric columns

numecic_columns = num_category_columns.copy()

for col in train_df.columns[:-1]:

    if col not in category_columns:

        numecic_columns.append(col)

numecic_columns
#LabelEncoding for numerical category columns

from sklearn.preprocessing import LabelEncoder

for col in num_category_columns:

    le = LabelEncoder()

    le.fit(np.unique(train_df[col].fillna('NA').unique().tolist()+test_df[col].fillna('NA').unique().tolist()))

    train_df[col] = le.transform(train_df[col].fillna('NA'))

    test_df[col] = le.transform(test_df[col].fillna('NA'))
stats.skew(train_df[numecic_columns].dropna())
#Skewed Columns

skewed_features = train_df[numecic_columns].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

skew_df = pd.DataFrame(skewed_features, columns=['Skew'])

skew_df.head()
#歪度を直して正規分布に近づける

skew_df = skew_df[abs(skew_df) > 0.75]

print('There are {} skewed numerical features to Box Cos transform.'.format(skew_df.shape[0]))



from scipy.special import boxcox1p

lam = 0.15

for col in skew_df.index:

    train_df[col] = boxcox1p(train_df[col], lam)

    test_df[col] = boxcox1p(test_df[col], lam)
#colsはLabel Encodingが必要なもの。すなわち、文字列形式のカラムのもの。

#category扱いのカラムでなおかつ、

cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 

       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 

       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 

       'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 

       'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']



for col in cols:

    if col not in numecic_columns:

        le = LabelEncoder()

        le.fit(np.unique(train_df[col].fillna('NA').unique().tolist() + test_df[col].fillna('NA').unique().tolist()))

        

        print(col)

        train_df[col] = le.transform(train_df[col].fillna('NA'))

        test_df[col] = le.transform(test_df[col].fillna('NA'))
#トレーニングデータとテストデータを作成

X_train = train_df.drop(['Id', 'SalePrice'], axis=1)

y_train = train_df[['SalePrice']]

X_test = test_df.drop('Id', axis=1)
from xgboost import XGBRegressor
model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model.fit(X_train, y_train)

predict = np.expm1(model.predict(X_test))
#np.log1pでlog(x+1)の変換をしたので、予測結果を逆関数で戻す

#predict_updated = np.expm1(predict)
submit_df['SalePrice'] = predict
submit_df.to_csv('result.csv', index=False)