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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.iloc[:5,:20]
train.iloc[:,:20].dtypes
train.iloc[:5,20:40]
train.iloc[:,20:40].dtypes
train.iloc[:5,40:60]
train.iloc[:,40:60].dtypes
train.iloc[:5,60:80]
train.iloc[:,60:80].dtypes
import seaborn as sns
sns.regplot(x=train['YearBuilt'],y=train['SalePrice'])
sns.regplot(x=train['YearRemodAdd'],y=train['SalePrice'])
sns.regplot(x=train['OverallQual'],y=train['SalePrice'])
sns.swarmplot(x=train['Neighborhood'],y=train['SalePrice'])
sns.barplot(x=train['Condition1'],y=train['SalePrice'])
#'YrSold', 'MoSold'
sns.swarmplot(x=train['Utilities'],y=train['SalePrice'])
sns.swarmplot(x=train['LandSlope'],y=train['SalePrice'])
sns.swarmplot(x=train['RoofStyle'],y=train['SalePrice'])
sns.swarmplot(x=train['OverallCond'],y=train['SalePrice'])
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
sns.barplot(x = train['PoolArea'], y = train['SalePrice'])
plt.figure(figsize=(20,6))
sns.regplot(x = train['YearBuilt'], y = train['SalePrice'])
train.YearBuilt.unique()
def get_data_split(dataframe, split_num = 0.2):
    split_num = 0.2
    split_point = int(len(dataframe)*split_num)
    train_x = dataframe[:-split_point]
    val_x = dataframe[-split_point:]
    return train_x, val_x
    
#missing_cols = [col for col in train_x.columns if train_x[col].isnull().any()] 에서 영감을 받음
#다음번에도 쓸 수 있게 나만의 함수로 제작(missing value가 있는 columns만 추려내줌)

def find_missing_cols(dataframe, list):
    list = []
    columns = dataframe.columns
    for col in columns:
        missing_judgement = dataframe[col].isnull().any()
        if missing_judgement == True:
            list.append(col)
        else:
            pass
    return list
train_x, val_x = get_data_split(train)
missing_col=[]
missing_col = find_missing_cols(train_x,missing_col)
first_train_x = train_x.drop(missing_col, axis=1)
first_test = test.drop(missing_col, axis=1)
first_col = first_train_x.columns
def find_category_col(dataframe):
    s = (dataframe.dtypes == 'object')
    object_cols = list(s[s].index)
    return object_cols

first_cat_col = find_category_col(first_train_x)
test_cat_col = find_category_col(first_test)
len(first_cat_col)
import category_encoders as ce
cat_features = first_cat_col
test_cat_features = test_cat_col

count_enc = ce.CountEncoder()

count_encoded = count_enc.fit_transform(first_train_x[cat_features])
test_count_encoded = count_enc.fit_transform(first_test[test_cat_features])
#first_train_x = first_train_x.join(count_encoded.add_suffix('_count'))
#first_train_x.drop(first_train_x[cat_features])
new_first_x = first_train_x.join(count_encoded.add_suffix('_counted'))
new_test = first_test.join(test_count_encoded.add_suffix('counted'))
new_first_x = new_first_x.drop(cat_features, axis=1)
new_test_first_x = new_test.drop(test_cat_features, axis=1)
new_first_x
new_test_first_x
new_first_x.columns
from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = new_first_x.columns.drop('SalePrice')

selector = SelectKBest(f_classif, k=30)

x_new_1 = selector.fit_transform(new_first_x[feature_cols],
                                new_first_x['SalePrice'])

x_new_1
first_col = ['LotArea', 'OverallQual','1stFlrSF', '2ndFlrSF', 'HalfBath', 'BedroomAbvGr',
             'KitchenAbvGr','Fireplaces','GarageArea','PoolArea','SalePrice']
first_col_test = ['LotArea', 'OverallQual','1stFlrSF', '2ndFlrSF', 'HalfBath', 'BedroomAbvGr',
             'KitchenAbvGr','Fireplaces','GarageArea','PoolArea']
new_first_x
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
sns.scatterplot(x=train['1stFlrSF'],y=train['SalePrice'])
sns.scatterplot(x=train['2ndFlrSF'],y=train['SalePrice'])
len(train['2ndFlrSF'])
train['LowQualFinSF'].value_counts()
train['GrLivArea'].value_counts()
plt.figure(figsize=(20,10))
sns.scatterplot(x = train['GrLivArea'], y=train['SalePrice'], hue=train['2ndFlrSF'])
train['FullBath'].hist()
sns.barplot(x=train['FullBath'],y=train['SalePrice'])
train['HalfBath'].hist()
sns.barplot(x=train['HalfBath'], y=train['SalePrice'])
train['BedroomAbvGr'].value_counts()
#sns.scatterplot(x=train['BedroomAbvGr'], y=train['SalePrice'])
sns.kdeplot(data=train['BedroomAbvGr'], shade=True)
#sns.regplot(x=train['BedroomAbvGr'], y=train['SalePrice'])
train['KitchenAbvGr'].value_counts()
train['KitchenQual'].value_counts()
from sklearn.preprocessing import OneHotEncoder
kitchenqual = pd.get_dummies(train['KitchenQual'])

train['Fireplaces'].value_counts()

sns.barplot(x=train['Fireplaces'], y=train['SalePrice'])
train['PoolArea'].value_counts()
sns.scatterplot(x=train['PoolArea'], y=train['SalePrice'])
def scatterplot (x,y):
    plt.figure(figsize=(16,6))
    sns.scatterplot(x=train[x], y=y)
def barplot (x,y):
    plt.figure(figsize=(16,6))
    sns.barplot(x=train[x], y=y)
def regplot(x,y):
    plt.figure(figsize=(16,6))
    sns.regplot(x=train[x], y=y)
def lineplot(x,y):
    plt.figure(figsize=(16,6))
    sns.lineplot(x=train[x], y=y)
train.groupby('MasVnrType').MasVnrType.size()
sns.barplot(x = train.MasVnrType, y = train.SalePrice)
plt.figure(figsize = (18,8))
sns.lineplot(x = train.MasVnrArea, y = train.SalePrice)
plt.figure(figsize = (18,8))
sns.lmplot(x = 'MasVnrArea', y = 'SalePrice', hue = 'MasVnrType', data=train)
###MasVnrType은 None으로, MasVnrArea는 0으로 채워넣자
train.MasVnrType = train.MasVnrType.fillna('None')
test.MasVnrType = test.MasVnrType.fillna('None')
#train.MasVnrArea = train.MasVnrArea.fillna(0)
#test.MasVnrArea = test.MasVnrArea.fillna(0)
bsmt_features_not_type = [ 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
bsmt_features_type = ['BsmtQual','BsmtCond', 'BsmtExposure',
              'BsmtFinType1', 'BsmtFinType2','BsmtFullBath','BsmtHalfBath']
#Bath는 type은 아니지만 몇 개로 분류해서 type으로 구분
for bsmt in bsmt_features_not_type:
    regplot(bsmt,y=train.SalePrice)
for bsmt in bsmt_features_type:
    barplot(bsmt, y=train.SalePrice)
train.loc[train.BsmtFullBath == 3]
### 지하실 관련된 것들은 다 'NA'로 바꾸어준다. 어짜피 이에 따른 값들은 이미 0임! 새로운 규칙 발견 예상
for bsmt in ['BsmtQual','BsmtCond', 'BsmtExposure',
              'BsmtFinType1', 'BsmtFinType2','BsmtFullBath']:
    train[bsmt] = train[bsmt].fillna('NA')
    test[bsmt] = test[bsmt].fillna('NA')
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
for bsmt in bsmt_features_type:
    plt.figure(figsize=(16,6))
    sns.barplot(x=train[bsmt], y=train.SalePrice)
garage_features_not_type = ['GarageArea', 'GarageYrBlt']
garage_features_type = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars']
for gg in garage_features_not_type:
    regplot(gg,train.SalePrice)
for gg in garage_features_type:
    barplot(gg,train.SalePrice)
round(train.GarageYrBlt.mean())
for gg in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[gg] = train[gg].fillna('NA')
    test[gg] = test[gg].fillna('NA')
#train.GarageYrBlt = train.GarageYrBlt.fillna(round(train.GarageYrBlt.mean()))
#test.GarageYrBlt = test.GarageYrBlt.fillna(1979)
test['GarageArea'] = test['GarageArea'].fillna(0)
test['GarageCars'] = test['GarageCars'].fillna(0)
lot_features = ['LotArea', 'LotShape', 'LotConfig']
lot_features_type = ['LotShape', 'LotConfig']
lot_features_not_type = 'LotArea'
for lot in lot_features_type:
    barplot(lot,y=train.SalePrice)
lineplot('LotArea',y=train.SalePrice)
regplot('LotArea',y=train.SalePrice)
sns.scatterplot(train.MiscVal,y=train.SalePrice)
train.SalePrice.mean()
y=train.SalePrice
plt.figure(figsize=(16,6))
sns.regplot(x=train.MiscVal, y=y)
pd.DataFrame({"Year" : train.YrSold, "Month" : train.MoSold})
sns.barplot(train.YrSold,y=train.SalePrice)
sns.barplot(train.MoSold,y=train.SalePrice)
date = train.groupby(['YrSold', 'MoSold']).SalePrice.mean()
sns.lineplot(x=range(55) ,y=date.values.tolist())
train['dates'] = train.YrSold.astype(str) + '-' + train.MoSold.astype(str)
test['dates'] = test.YrSold.astype(str) + '-' + test.MoSold.astype(str)
sns.barplot(train.dates,y=train.SalePrice)
#마지막에 X로 drop시켜서 X 만들기
X = train.drop('BsmtFullBath', axis = 1)
test= test.drop('BsmtFullBath', axis = 1)
cat_features = X.select_dtypes(include=['object']).columns
cat_features
train.shape
test.shape
for col in train.columns:
    if not col in test.columns:
        print(col)
for col in test.columns:
    if not col in train_X.columns:
        print(col)
import category_encoders as ce
target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(X, y)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.columns
features=['YearBuilt', 'YearRemodAdd', 'OverallQual',
'1stFlrSF', '2ndFlrSF', 'GrLivArea','Fireplaces','BedroomAbvGr',
'MasVnrType', 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtQual','BsmtCond', 'BsmtExposure',
              'BsmtFinType1', 'BsmtFinType2','BsmtFullBath', 
'GarageArea', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars', 'LotArea',
'LotShape' ,'YrSold', 'MoSold', 'HalfBath']
train_X=train[features]
train_Y=train['SalePrice']
X_test=test[features]

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=2020, shuffle=True)
cat_features = [col for col in X_train.columns if X_train[col].dtype=='object']
num_features = [col for col in X_train.columns if X_train[col].dtype in ['int64','float64']]
len(cat_features), len(num_features), len(X_train.columns)
for col in cat_features:
    val = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(val)
    X_valid[col] = X_valid[col].fillna(val)
    X_test[col] = X_test[col].fillna(val)
    
for col in num_features:
    val = X_train[col].mean()
    X_train[col] = X_train[col].fillna(val)
    X_valid[col] = X_valid[col].fillna(val)
    X_test[col] = X_test[col].fillna(val)
X_train.isnull().sum().sort_values(ascending=False)
y_train.isnull().sum()
from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in cat_features:
    encoder = LabelEncoder()
    X_ = pd.concat([X_train[col],X_valid[col]],axis=0)
    encoder.fit(X_)
    X_train[col] = encoder.transform(X_train[col])
    X_valid[col] = encoder.transform(X_valid[col])
    encoders[col] = encoder
for col in cat_features:
    encoder = encoders[col]
    encoder.transform(X_test[col])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(y,preds):
    return sqrt(mean_squared_error(y, preds))
y_train = np.log(y_train)
y_valid = np.log(y_valid)
rf = RandomForestRegressor()
#X_train = X_train.drop('BsmtFullBath', axis = 1)
#X_valid = X_valid.drop('BsmtFullBath', axis = 1)
#test= test.drop('BsmtFullBath', axis = 1)
rf.fit(X_train,y_train)
preds = rf.predict(X_valid)
rmse(y_valid,preds)
tmp = pd.DataFrame({'importance':rf.feature_importances_,'feature':X_train.columns}).sort_values('importance', ascending=False)
tmp.iloc[:30]
train_X.dtypes
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': rf.predict(test)})
output.to_csv('submission.csv', index=False)
