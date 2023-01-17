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
import matplotlib.pyplot as plt 
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
def getNullsCols(data):
    drop_cols = []
    for col in data.columns :
        nans = data[col].isna().sum()
        if nans > len(data)*0.30:
            drop_cols.append(col)
    return drop_cols
train_dr_cols = getNullsCols(train)
test_dr_cols = getNullsCols(test)
train_dr_cols, test_dr_cols
#dropping the columns as the have the null values more than 30 %

train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace = True, axis = 1)
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace = True, axis = 1)
train.head()
##Filling Na 
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
digitcols_data_train = train.select_dtypes(numerics).columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
digitcols_data_test = test.select_dtypes(numerics).columns
digitcols_data_train, digitcols_data_test


len(digitcols_data_train),len(digitcols_data_test)
def fillNaForNumerCols(data, cols):
    for col in cols :
        if data[col].isna().sum() > 0.0 :
            print(data[col].mean())
            data[col].fillna(data[col].mean(), inplace = True)
    return data
trian = fillNaForNumerCols(train, digitcols_data_test)
train[digitcols_data_test].isna().sum()
test = fillNaForNumerCols(test, digitcols_data_test)
test[digitcols_data_test].isna().sum()
def fillNaForObjCols(data):
    obj_cols = data.select_dtypes(['object','category'])
    for col in obj_cols:
        if data[col].isna().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace = True)
    return data
train_t
train_t = fillNaForObjCols(train)
test_t = fillNaForObjCols(train)
for i in train_t.isna().sum() > 0:
    if i == True : 
        print(i)
for i in test_t.isna().sum() > 0:
    if i == True : 
        print(i)
test_t = fillNaForObjCols(test)
train = train_t
test = test_t
train.head()
test.head()
def getNonUnqCols(data, thresh):
    obj_cols = data.select_dtypes(['object', 'category'])
    unq = []
    for i in obj_cols:
        unq_no = len(data[i].unique())
        if unq_no > thresh:
            unq.append(i)
        
    return unq

nonUnq_train = getNonUnqCols(train, 6)
nonUnq_test = getNonUnqCols(test, 6)
nonUnq_train, nonUnq_test
nonUnq_train = getNonUnqCols(train, 9)
nonUnq_test = getNonUnqCols(test, 9)
nonUnq_train, nonUnq_test
nonUnq_train = getNonUnqCols(train, 5)
nonUnq_test = getNonUnqCols(test, 5)
nonUnq_train, nonUnq_test
#The condition2 doesnt even have the same uniq values in the test data , i dont think its usefull therefore drop it 
#Drop = ['Condition2']
#dropping the columns as the have the null values more than 30 %

train.drop(['Condition2'], inplace = True, axis = 1)
test.drop(['Condition2'], inplace = True, axis = 1)
nonUnq_train = getNonUnqCols(train, 7)
nonUnq_test = getNonUnqCols(test, 7)
nonUnq_train, nonUnq_test
#dropping the columns as the have the null values more than 30 %

train.drop(['HouseStyle'], inplace = True, axis = 1)
test.drop(['HouseStyle'], inplace = True, axis = 1)
#dropping the columns as the have the null values more than 30 %

train.drop(['RoofMatl'], inplace = True, axis = 1)
test.drop(['RoofMatl'], inplace = True, axis = 1)
nonUnq_train = getNonUnqCols(train, 7)
nonUnq_test = getNonUnqCols(test, 7)
nonUnq_train, nonUnq_test
nonUnq_train = getNonUnqCols(train, 10)
nonUnq_test = getNonUnqCols(test, 10)
nonUnq_train, nonUnq_test
#how  the cols 'Condition1', 'SaleType' are affecting the SalePrice
#drop the cols more than 10 distinct values 
#dropping the columns as the have the null values more than 30 %

train.drop(['Neighborhood', 'Exterior1st', 'Exterior2nd'], inplace = True, axis = 1)
test.drop(['Neighborhood', 'Exterior1st', 'Exterior2nd'], inplace = True, axis = 1)
import seaborn as sns
sns.catplot(x = 'Condition1', y = 'SalePrice', data = train)
train['Condition1'].unique()
sum(train['Condition1'] == 'RRNe'), sum(train['Condition1'] == 'PosA'), 
sum(train['Condition1'] == 'RRNn'), sum(train['Condition1'] == 'RRAe')
train.replace({'Condition1': {'RRNe': 'condLow', 'PosA': 'condLow'}}, inplace= True)
train.replace({'Condition1': {'RRAe': 'condLow'}}, inplace= True)
test.replace({'Condition1': {'RRNe': 'condLow', 'PosA': 'condLow','RRAe': 'condLow'}}, inplace= True)
train['Condition1'].unique()
test['Condition1'].unique()

sns.catplot(x = 'SaleType', y = 'SalePrice', data = train)
train['SaleType'].unique()
#make the cols :- Con, CWD  as SaleType_1
               # 'Oth', 'ConLw','ConLI','ConLD'   as SaleType_2 
train.replace({'SaleType': {'Con': 'SaleType_1', 'CWD': 'SaleType_1'}}, inplace= True)
test.replace({'SaleType': {'Con': 'SaleType_1', 'CWD': 'SaleType_1'}}, inplace= True)
train.replace({'SaleType': {'Oth': 'SaleType_2', 'ConLw': 'SaleType_2',
                            'ConLI': 'SaleType_2','ConLD': 'SaleType_2'}}, inplace= True)
test.replace({'SaleType': {'Oth': 'SaleType_2', 'ConLw': 'SaleType_2',
                            'ConLI': 'SaleType_2','ConLD': 'SaleType_2'}}, inplace= True)
train['SaleType'].unique()
test['SaleType'].unique()
#dropping the high correlated values 
train.corr()['SalePrice']
#dropping the OverallQual as it has close 80 % correlated to the SalePrice 
#dropping the columns as the have the null values more than 30 %

train.drop(['OverallQual'], inplace = True, axis = 1)
test.drop(['OverallQual'], inplace = True, axis = 1)
train.head()
test.head()
#Seems like the data has lot of zeros in it 
sum(train['EnclosedPorch'] == 0)
sum(test['EnclosedPorch'] == 0)
len(train)*0.30
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
check_zero_cols = test.select_dtypes(numerics).columns
for col in check_zero_cols:
    l = len(train)
    if sum(train[col] == 0) > (l*0.40):
        print(col)
for col in check_zero_cols:
    l = len(test)
    if sum(test[col] == 0) > (l*0.40):
        print('\''+col+'\',')
#dropping the columns as the have the null values more than 30 %

train.drop(['MasVnrArea',
'BsmtFinSF2',
'2ndFlrSF',
'LowQualFinSF',
'BsmtFullBath',
'BsmtHalfBath',
'HalfBath',
'Fireplaces',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal'], inplace = True, axis = 1)
test.drop(['MasVnrArea',
'BsmtFinSF2',
'2ndFlrSF',
'LowQualFinSF',
'BsmtFullBath',
'BsmtHalfBath',
'HalfBath',
'Fireplaces',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal'], inplace = True, axis = 1)
train.head()
test.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
check_zero_cols = test.select_dtypes(numerics).columns
for col in check_zero_cols:
    l = len(train)
    if sum(train[col] == 0) < (l*0.40) and sum(train[col] == 0) > (l*0.10):
        print('\''+col+'\',')
mean_train_BSMT  = train['BsmtFinSF1'].mean()
mean_test_BSMT  = test['BsmtFinSF1'].mean()
train.replace({'BsmtFinSF1': {0: mean_train_BSMT}}, inplace= True)
test.replace({'BsmtFinSF1': {0: mean_test_BSMT}}, inplace= True)
train.head()
test.head()
for col in check_zero_cols:
    l = len(train)
    if sum(train[col] == 0) < (l*0.40) and sum(train[col] == 0) > 0:
        print('\''+col+'\',')
sum(train['TotalBsmtSF'] ==0)
plt.scatter(train['TotalBsmtSF'] , train['SalePrice'] )
plt.scatter(train['FullBath'] , train['SalePrice'] )
plt.scatter(train['BsmtUnfSF'] , train['SalePrice'] )
plt.scatter(train['BedroomAbvGr'] , train['SalePrice'] )
plt.scatter(train['KitchenAbvGr'] , train['SalePrice'] )
plt.scatter(train['GarageCars'] , train['SalePrice'] )
plt.scatter(train['GarageArea'] , train['SalePrice'] )
#Change the Dtypes of the Classifies columns such as 
# FullBath, GarageCars, KitchenAbvGr, BedroomAbvGr 
train['FullBath'] = train['FullBath'].astype('object')
train['GarageCars'] = train['GarageCars'].astype('object')
train['KitchenAbvGr'] = train['KitchenAbvGr'].astype('object')
train['BedroomAbvGr'] = train['BedroomAbvGr'].astype('object')
test['FullBath'] = test['FullBath'].astype('object')
test['GarageCars'] = test['GarageCars'].astype('object')
test['KitchenAbvGr'] = test['KitchenAbvGr'].astype('object')
test['BedroomAbvGr'] = test['BedroomAbvGr'].astype('object')
#Ajust some mean value in the 'GarageArea' as it is separated from the other values 
#adjust to the rangeof 200 to 400 like 260 which gives the same result as for the 0 Garbase area 
train.replace({'GarageArea': {0: 260}}, inplace= True)
test.replace({'GarageArea': {0: 260}}, inplace= True)
X  = train.iloc[:, :-1]
X.head()
Y = pd.DataFrame(train.iloc[:, -1], columns={'SalePrice'})
Y
X.drop(['Id'], inplace=True, axis = 1)
test_ids = pd.DataFrame(test['Id'], columns={'Id'})
test.drop(['Id'], inplace=True, axis = 1)
X_dumm = pd.get_dummies(X, drop_first=True)
X_dumm
X_dumm.columns.values
test_dumm = pd.get_dummies(test, drop_first=True)
test_dumm
test_dumm.columns.values
for cols in X_dumm.columns.values:
    if cols not in test_dumm.columns.values:
        print(cols)
for cols in test_dumm.columns.values:
    if cols not in X_dumm.columns.values:
        print(cols)
#drop column in the test data are 
#'FullBath_4', 'GarageCars_1.7661179698216736', 'GarageCars_5.0'
test_dumm.drop(['FullBath_4', 'GarageCars_1.7661179698216736', 'GarageCars_5.0'], inplace = True, axis = 1)
for cols in test_dumm.columns.values:
    if cols not in X_dumm.columns.values:
        print(cols)
test_dumm.rename(columns={"GarageCars_1.0": "GarageCars_1", "GarageCars_2.0": "GarageCars_2",
                         "GarageCars_3.0": "GarageCars_3","GarageCars_4.0": "GarageCars_4"}, inplace=True)
for cols in test_dumm.columns.values:
    if cols not in X_dumm.columns.values:
        print(cols)
for cols in X_dumm.columns.values:
    if cols not in test_dumm.columns.values:
        print('\''+cols+'\',')
#dropping these are they are not in the test 
#'Utilities_NoSeWa',
'Heating_GasA',
'Heating_OthW',
'Electrical_Mix',
'BedroomAbvGr_8',
'KitchenAbvGr_3',
'GarageQual_Fa',
X_dumm.drop(['Utilities_NoSeWa',
'Heating_GasA',
'Heating_OthW',
'Electrical_Mix',
'BedroomAbvGr_8',
'KitchenAbvGr_3',
'GarageQual_Fa'], inplace = True, axis = 1)
X_dumm.head()
test_dumm.head()
for cols in X_dumm.columns.values:
    if cols not in test_dumm.columns.values:
        print('\''+cols+'\',')
#All the coolumns have become equal and ready to model 

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
boosting = GradientBoostingRegressor()
params = {
    'criterion':['mse'],
    "n_estimators" : [50,100,200,400,350 ],
    "max_depth" : [ 4,6]
}
cv = GridSearchCV(boosting, params, cv = 10)
cv.fit(X_dumm, Y)
boost_res = cv.predict(test_dumm)
final_deep = pd.concat([test_ids, pd.DataFrame(boost_res, columns={'SalePrice'})], axis=1)
final_deep.to_csv('/kaggle/input/house-prices-advanced-regression-techniques/submission.csv', index=False)
