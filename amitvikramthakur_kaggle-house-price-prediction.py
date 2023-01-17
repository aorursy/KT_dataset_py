import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
#import missingno as msno
#from sklearn.ensemble import RandomForestRegressor

dfTrain = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",encoding="utf-8")
dfTest = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",encoding="utf-8")
dfSampleSubmission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv",encoding="utf-8")
pd.set_option('display.max_columns', None)
dfTrain.head()
dfTest.head()
dfSampleSubmission.head()
#msno.matrix(df)
dfTrain.shape
dfTrain.columns
featuresCategorical = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 
    'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
    'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',  'Functional', 
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
    'PavedDrive',  'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 
    'SaleType', 'SaleCondition'
]

featuresCategoricalExtra = ['Neighborhood']

featuresNumerical = [
    'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
    'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 
    'YrSold'
]

targerVariable = 'SalePrice'
dfTrain = dfTrain.fillna("NAN_AKLJHGJKADHFKLGHIUYEIRTH")
dfTest = dfTest.fillna("NAN_AKLJHGJKADHFKLGHIUYEIRTH")
testRow = 0
condition = np.array([True]*len(dfTrain));
for fC in featuresCategorical:
    if (condition == True).sum() <= 1:
        break
    condition = condition & (dfTrain[fC] == dfTest[testRow:testRow+1].head(1)[fC].values[0])

dfTrain[condition].head()
#dfTrain[(dfTrain['Alley'] == undefine)].head()
#dfTrain['Alley'].unique()
featuresCategorical[0:11]
dfTest[0:1].head(1)[featuresCategorical[0]].values[0]
x = np.arange(len(dfTrain))
y = dfTrain['Id'].values
plt.plot(x,y,'.')
plt.show()
df = df.drop(columns=['Id'])
df.head()
features_numerical = np.setdiff1d(df.columns,np.array(features_categorical))
df2 = pd.get_dummies(df, columns=features_categorical)
df2.head()
df2.shape
msno.matrix(df2[features_numerical])
msno.matrix(df2.drop(columns=features_numerical))
# to interpolate the missing values  
df2[features_numerical] = df2[features_numerical].interpolate(method ='linear', limit_direction ='forward') 
# to interpolate the missing values  
df2[features_numerical] = df2[features_numerical].interpolate(method ='linear', limit_direction ='backward') 
msno.matrix(df2)
train_X = df2.drop(columns=['SalePrice'])[ : int(0.7*len(df2)) ]
train_y = df2['SalePrice'].values[ : int(0.7*len(df2)) ]
test_X = df2.drop(columns=['SalePrice'])[ int(0.7*len(df2)) : ]
test_y = df2['SalePrice'].values[ int(0.7*len(df2)) : ]
train_X.head()
model_RandomForestRegressor = RandomForestRegressor()
model_RandomForestRegressor.fit(train_X,train_y)
print(model_RandomForestRegressor.score(train_X,train_y))
print(model_RandomForestRegressor.score(test_X,test_y))
df_submission_test_original = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",encoding="utf-8")
df_submission_test = df_submission_test_original.copy()
df_submission_test.head()
id_df_submission_test = df_submission_test['Id'].values
msno.matrix(df_submission_test)
df_submission_test.shape
df_submission_test = df_submission_test.drop(columns=['Id'])
df_submission_test2 = pd.get_dummies(df_submission_test, columns=features_categorical)
df_submission_test2.shape
features_numerical_test = np.setdiff1d(features_numerical,np.array(['SalePrice']))
msno.matrix(df_submission_test2[features_numerical_test])
msno.matrix(df_submission_test2.drop(columns=features_numerical_test))
# to interpolate the missing values  
df_submission_test2[features_numerical_test] = df_submission_test2[features_numerical_test].interpolate(method ='linear', limit_direction ='forward') 
# to interpolate the missing values  
df_submission_test2[features_numerical_test] = df_submission_test2[features_numerical_test].interpolate(method ='linear', limit_direction ='backward') 
msno.matrix(df_submission_test2)
print(len(df2.columns))
print(len(train_X.columns))
print(len(df_submission_test2.columns))
columns_test = np.union1d(df2.columns,df_submission_test2.columns)
print(len(columns_test))
df_submission_test3 = pd.DataFrame(data=df_submission_test2,columns=train_X.columns)
df_submission_test3.head()
df_submission_test3.shape
msno.matrix(df_submission_test3)
df_submission_test3 = df_submission_test3.fillna(0)
msno.matrix(df_submission_test3)
submission_y = model_RandomForestRegressor.predict(df_submission_test3)
submission_y
df_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv",encoding="utf-8")
df_submission.head()
df_submission['Id'] = id_df_submission_test
df_submission['SalePrice'] = submission_y
df_submission.head()
df_submission.shape
df_submission.to_csv("submission.csv",index=False)
