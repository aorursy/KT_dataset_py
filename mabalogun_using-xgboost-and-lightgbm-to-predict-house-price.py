# Import packages

import numpy as np

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import sklearn.metrics as metrics

import math

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



train.head()
train.drop('Id', axis= 1, inplace= False)



test.drop('Id', axis= 1, inplace= False)
train.shape
train.info()
train.describe(include='all')
test.head()
test.shape
train1= train

test1= test 


df = pd.concat([train1, test1], axis=0,sort=False)

df.shape
df.describe(include='all')
df.isnull().sum()
df["PoolQC"] = df["PoolQC"].fillna("None")



df["MiscFeature"] = df["MiscFeature"].fillna("None")



df["Alley"] = df["Alley"].fillna("None")



df['Fence'] = df['Fence'].fillna('None')



df['PoolQC'] = df['PoolQC'].fillna('None')



df['GarageCond'] = df['GarageCond'].fillna('None')



df['GarageQual'] = df['GarageQual'].fillna('None')



df['GarageFinish'] = df['GarageFinish'].fillna('None')



df['GarageType'] = df['GarageType'].fillna('None')



df['FireplaceQu'] = df['FireplaceQu'].fillna('None')



df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')



df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')



df['BsmtExposure'] = df['BsmtExposure'].fillna('None')



df['BsmtCond'] = df['BsmtCond'].fillna('None')



df['BsmtQual'] = df['BsmtQual'].fillna('None')
df.isnull().sum()
df_missing = (df.isnull().sum() / len(df)) * 100

df_missing = df_missing.drop(df_missing[df_missing == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df_missing})

missing_data.head(30)
df = df.fillna(df.median())

df_missing = (df.isnull().sum() / len(df)) * 100

df_missing = df_missing.drop(df_missing[df_missing == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df_missing})

missing_data.head(30)
df = df.fillna(df.median())
df = df.fillna(df.median())

df_missing = (df.isnull().sum() / len(df)) * 100

df_missing = df_missing.drop(df_missing[df_missing == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df_missing})

missing_data.head(30)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
df = df.fillna(df.median())

df_missing = (df.isnull().sum() / len(df)) * 100

df_missing = df_missing.drop(df_missing[df_missing == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df_missing})

missing_data.head(30)
object_columns_df = df.select_dtypes(include=['object'])

numerical_columns_df =df.select_dtypes(exclude=['object'])
object_columns_df.dtypes
object_columns_df.columns
numerical_columns_df.dtypes
corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
df= pd.get_dummies(df, columns= ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition'])
df.shape
object_columns_df['Street'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Street'].value_counts() 
object_columns_df['Condition2'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Condition2'].value_counts() 

object_columns_df['RoofMatl'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['RoofMatl'].value_counts() 
object_columns_df['Heating'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Heating'].value_counts() #======> Drop feature one Type

object_columns_df = object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)

numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

numerical_columns_df['Age_House'].describe()

Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]

Negatif

numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'],'YrSold' ] = 2009

numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

numerical_columns_df['Age_House'].describe()

numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath']*0.5

numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath']*0.5 

numerical_columns_df['TotalSA']=numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF']

numerical_columns_df.head()
bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,

            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6

            }

object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)

object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)

object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)

object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)

object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)

object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)

object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)

object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)

object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)

object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)

object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)

object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)

object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)

object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)



PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}

object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)



#Select categorical features

rest_object_columns = object_columns_df.select_dtypes(include=['object'])

#Using One hot encoder

object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns) 

object_columns_df.head()
df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)

df_final.head()
df_final.shape
df_final.columns
df_train = df_final.iloc[:1460,:]



df_test = df_final.iloc[1460:,:]



print("Shape of new dataframes - {} , {}".format(df_train.shape, df_test.shape)) 

target= df_train['SalePrice']

df_train = df_train.drop(['SalePrice'],axis=1)
df_train.shape
df_test = df_test.drop(['SalePrice'],axis=1)

df_test.head()
df_test.shape
x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.2,random_state=0)


xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=1.5, n_estimators=2400,

             n_jobs=1, nthread=None, objective='reg:linear',

             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

             silent=None, subsample=0.8, verbosity=1)





lgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=12000, 

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.4, 

                                       )

#Fitting

xgb.fit(x_train, y_train)

lgbm.fit(x_train, y_train,eval_metric='rmse')

predict1 = xgb.predict(x_test)

predict = lgbm.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

xgb.fit(df_train, target)

lgbm.fit(df_train, target,eval_metric='rmse')

predict4 = lgbm.predict(df_test)

predict3 = xgb.predict(df_test)

predict_y = ( predict3*0.575 + predict4 * 0.425)

submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict_y

    })

submission.to_csv('submission.csv', index=False)



print ('Submission completed')
