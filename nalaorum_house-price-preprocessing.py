import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train = train.drop(['Id'],axis=1)

# result에 id를 추가해야 하므로 따로 빼두자

test_ids = test['Id']

df_test = test.drop(['Id'],axis=1)
df_num_to_category_cols = ['MSSubClass', 'OverallCond', 'KitchenAbvGr', 'BedroomAbvGr','TotRmsAbvGrd']
df_train_numbers = df_train[df_train.columns[df_train.dtypes != object]]

df_train_numbers = df_train_numbers.drop(df_num_to_category_cols, axis=1)

df_test_numbers = df_test[df_test.columns[df_test.dtypes != object]]

df_test_numbers = df_test_numbers.drop(df_num_to_category_cols, axis=1)
df_train_categorys = df_train[df_train.columns[df_train.dtypes == object]].astype('category')

df_train_categorys = pd.concat([df_train_categorys,

                         df_train[df_num_to_category_cols]]

                        , axis=1)

df_train_categorys = df_train_categorys.astype('category')



df_test_categorys = df_test[df_test.columns[df_test.dtypes == object]].astype('category')

df_test_categorys = pd.concat([df_test_categorys,

                         df_test[df_num_to_category_cols]]

                        , axis=1)

df_test_categorys = df_test_categorys.astype('category')
df_train_numbers.isnull().sum()
df_test_numbers.isnull().sum()
# GarageYrBlt -> Yearbuilt 로 채우기

# MasVnrArea -> 0으로 채우기

# LotFrontage -> 평균으로 채우기

## test data는 아래 데이터도 NA가 있음 -> 모두 평균으로 채움

# BsmtFinSF1

# BsmtFinSF2

# BsmtUnfSF

# TotalBsmtSF

# BsmtFullBath

# BsmtHalfBath

# GarageCars

# GarageArea
year_built_list = df_train_numbers[df_train_numbers['GarageYrBlt'].isnull() ==True]['YearBuilt'].tolist()
year_built_list_2 = df_test_numbers[df_test_numbers['GarageYrBlt'].isnull() ==True]['YearBuilt'].tolist()
garage_list = []

i = 0

for v in df_train_numbers['GarageYrBlt'].values:

    if math.isnan(v):

        garage_list.append(year_built_list[i])

        i += 1

    else:

        garage_list.append(v)
garage_list_2 = []

i = 0

for v in df_test_numbers['GarageYrBlt'].values:

    if math.isnan(v):

        garage_list_2.append(year_built_list_2[i])

        i += 1

    else:

        garage_list_2.append(v)
df_train_numbers['GarageYrBlt'] = garage_list
df_test_numbers['GarageYrBlt'] = garage_list_2
df_train_numbers['GarageYrBlt'].isnull().sum()
df_test_numbers['GarageYrBlt'].isnull().sum()
df_train_numbers['MasVnrArea'] = df_train_numbers['MasVnrArea'].apply(lambda x: 0 if math.isnan(x) else x)
df_test_numbers['MasVnrArea'] = df_test_numbers['MasVnrArea'].apply(lambda x: 0 if math.isnan(x) else x)
df_train_numbers['MasVnrArea'].isnull().sum()
df_test_numbers['MasVnrArea'].isnull().sum()
df_train_numbers['LotFrontage'] = df_train_numbers['LotFrontage'].apply(lambda x: int(np.mean(df_train_numbers['LotFrontage'])) if math.isnan(x) else x)
df_test_numbers['LotFrontage'] = df_test_numbers['LotFrontage'].apply(lambda x: int(np.mean(df_test_numbers['LotFrontage'])) if math.isnan(x) else x)
df_train_numbers['LotFrontage'].isnull().sum()
df_test_numbers['LotFrontage'].isnull().sum()
df_test_numbers['BsmtHalfBath']
test_data_nan_cols = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
for col in test_data_nan_cols:

    df_test_numbers[col] = df_test_numbers[col].apply(lambda x: int(np.mean(df_test_numbers[col])) if math.isnan(x) else x)
df_train_categorys.isnull().sum()
df_test_categorys.isnull().sum()
# Alley, PoolQC, Fence, MiscFeature 삭제
df_train_categorys = df_train_categorys.drop(['Alley','PoolQC','Fence','MiscFeature'], axis=1)

df_test_categorys = df_test_categorys.drop(['Alley','PoolQC','Fence','MiscFeature'], axis=1)
# BsmtQual          37

# BsmtCond          37

# BsmtExposure      38

# BsmtFinType1      37

# BsmtFinType2      38

# GarageType        76

# GarageFinish      78

# GarageQual        78

# GarageCond        78

# FireplaceQu
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2

# 은 'No'로 채움
var_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

           'GarageType','GarageFinish','GarageQual','GarageCond','FireplaceQu']
for var in var_list:

    fill_no_list = []

    for v in df_train_categorys[var].values:

        if type(v) != str:

            fill_no_list.append('No')

        else:

            fill_no_list.append(v)

    df_train_categorys[var] = fill_no_list

    df_train_categorys[var] = df_train_categorys[var].astype('category')
for var in var_list:

    fill_no_list_2 = []

    for v in df_test_categorys[var].values:

        if type(v) != str:

            fill_no_list_2.append('No')

        else:

            fill_no_list_2.append(v)

    df_test_categorys[var] = fill_no_list_2

    df_test_categorys[var] = df_test_categorys[var].astype('category')
# Electrical nan 1개 -> 제일 많은 값 SBrkr로 채움
df_train_categorys['Electrical'][df_train_categorys['Electrical'].isnull() == True] = 'SBrkr'
# MasVnrType nan 8개 -> None으로 채움
df_train_categorys['MasVnrType'][df_train_categorys['MasVnrType'].isnull() == True] = 'None'
# test data 남은 nan data 처리 

# MSZoning          4  -> OverallQual에 따라 ['C (all)','C (all)','RL','C (all)'] 로 입력

# Utilities         2 - > AllPub 으로 채움

# Exterior1st       1 -> VinylSd

# Exterior2nd       1 -> VinylSd

# MasVnrType       16 -> 6이상은 stone, 4는 BrkCmn

# KitchenQual       1 -> TA

# Functional        2 -> Typ

# SaleType          1 -> WD
#df_test_categorys['SaleType'].hist()
df_test_categorys['MSZoning'][df_test_categorys['MSZoning'].isnull() == True] = ['C (all)','C (all)','RL','C (all)']
df_test_categorys['Utilities'][df_test_categorys['Utilities'].isnull() == True] = 'AllPub'
df_test_categorys['Exterior1st'][df_test_categorys['Exterior1st'].isnull() == True] = 'VinylSd'
df_test_categorys['Exterior2nd'][df_test_categorys['Exterior2nd'].isnull() == True] = 'VinylSd'
MasVnrType_list = []

for v in df_test_numbers['OverallQual'][df_test_categorys['MasVnrType'].isnull() == True].values:

    if v > 5:

        MasVnrType_list.append('Stone')

    else:

        MasVnrType_list.append('BrkCmn')
df_test_categorys['MasVnrType'][df_test_categorys['MasVnrType'].isnull() == True] = MasVnrType_list
df_test_categorys['KitchenQual'][df_test_categorys['KitchenQual'].isnull() == True] = 'TA'
df_test_categorys['Functional'][df_test_categorys['Functional'].isnull() == True] = 'Typ'
df_test_categorys['SaleType'][df_test_categorys['SaleType'].isnull() == True] = 'WD'
new_train_data = pd.concat([df_train_numbers, df_train_categorys], axis=1)

new_test_data = pd.concat([df_test_numbers, df_test_categorys], axis=1)
new_train_data = pd.get_dummies(new_train_data)

new_test_data = pd.get_dummies(new_test_data)
empty_cols = []

for col in new_train_data.columns:

    if col not in new_test_data.columns:

        empty_cols.append(col)
empty_cols[1:]
for col in empty_cols[1:]:

    new_test_data[col] = 0
empty_cols = []

for col in new_test_data.columns:

    if col not in new_train_data.columns:

        empty_cols.append(col)
empty_cols
for col in empty_cols:

    new_train_data[col] = 0
new_train_data.info()
new_test_data.info()
# train data에서 outlier 제거

plt.scatter(new_train_data['SalePrice'], new_train_data['PoolArea'])
new_train_data = new_train_data.drop(new_train_data.sort_values('GrLivArea', ascending=False)[:2]['GrLivArea'].index)
full_train_data = new_train_data.copy()

full_test_data = new_test_data.copy()
# SalePrice 제거 및 별도로 만들기

target_data = full_train_data['SalePrice']

full_train_data = full_train_data.drop('SalePrice', axis=1)
full_train_data.info()
len(target_data)
full_test_data.info()
# train = full_train_data

# test = full_test_data

# target = target_data

# test_id = test_ids
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(full_train_data, target_data, random_state=12)
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test,y_test)
result = rf.predict(full_test_data)
feature_importance_df = pd.DataFrame({'name':X_train.columns,'score':rf.feature_importances_})
ic = feature_importance_df.sort_values('score',ascending=False)['name'].head(160).values
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
scaler = StandardScaler()

scale_X_data = scaler.fit_transform(full_train_data)
pca1 = PCA(n_components=3)



principalComponents2 = pca1.fit_transform(scale_X_data)

principalDf2 = pd.DataFrame(data = principalComponents2

             , columns = ['principal component '+str(i) for i in range(1,4)])

finalDataFrame2 = pd.concat([principalDf2, pd.DataFrame({'labels':target_data.values})], axis=1)
#%matplotlib widget
#%matplotlib inline
fig2 = plt.figure(figsize=(30,30))

ax2 = fig2.add_subplot(111, projection='3d')



ax2.set_xlabel('Principal Component 1', fontsize = 15)

ax2.set_ylabel('Principal Component 2', fontsize = 15)

ax2.set_zlabel('Principal Component 3', fontsize = 15)

ax2.set_title('3 Component PCA', fontsize = 20)        



xAxisLine = ((min(principalDf2['principal component 1']), max(principalDf2['principal component 1'])), (0, 0), (0,0))

ax2.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')

yAxisLine = ((0, 0), (min(principalDf2['principal component 2']), max(principalDf2['principal component 2'])), (0,0))

ax2.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'b')

zAxisLine = ((0, 0), (0,0), (min(principalDf2['principal component 3']), max(principalDf2['principal component 3'])))

ax2.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'g')  



ax2.scatter(principalDf2['principal component 1'],

           principalDf2['principal component 2'],

           principalDf2['principal component 3'])
pca1.explained_variance_ratio_
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge.score(X_train,y_train)
ridge.score(X_test,y_test)
result = ridge.predict(full_test_data)
df_01 = pd.DataFrame({'Id':test_ids, 'SalePrice':result})

df_01.to_csv('submission_xgb_1220_2.csv', index=False)
result
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,

                    learning_rate=0.01574, max_depth=3,

                    min_child_weight=1.7817, n_estimators=2200,

                    reg_alpha=0.46117, reg_lambda= 0.649,

                    subsample=0.5213, silent=1,

                    objective ='reg:squarederror',

                    random_state=7, nthread=-1)
model_xgb.fit(X_train[ic], y_train)
model_xgb.score(X_test[ic], y_test)
result = model_xgb.predict(full_test_data[ic])