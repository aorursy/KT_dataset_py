# Import packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import sklearn.metrics as metrics

import math

import mpl_toolkits

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)

%matplotlib inline 
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#Creating a copy of the train and test datasets

test1  = test.copy()

train1  = train.copy()
test1.head()
train1= train1[pd.notnull(train1['SalePrice'])]

train1.tail()
train1.head()
train1['train']  = 1

test1['train']  = 0

house = pd.concat([train1,test1], axis=0,sort=False)

house.tail()
plt.scatter(house.MSSubClass,house.SalePrice)

plt.title('Price vs Mssubclass')
plt.scatter(house.LotFrontage,house.SalePrice)

plt.title('Price vs LotFrontage')
plt.scatter(house.LotArea,house.SalePrice)

plt.title('Price vs LotArea')
plt.scatter(house.Street,house.SalePrice)

plt.title('Price vs Street')
plt.scatter(house.LotShape,house.SalePrice)

plt.title('Price vs LotShape')
plt.scatter(house.LandContour,house.SalePrice)

plt.title('Price vs LandContour')
#Percentage of NAN Values 

NAN = [(c, house[c].isna().mean()*100) for c in house]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN = NAN[NAN.percentage > 50]

NAN.sort_values("percentage", ascending=False)
house = house.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
object_columns_df = house.select_dtypes(include=['object'])

numerical_columns_df =house.select_dtypes(exclude=['object'])
object_columns_df.dtypes
numerical_columns_df.dtypes
#Number of null values in each feature

null_counts = object_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
columns_None = ['FireplaceQu']

object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')
#'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual',,'GarageCond'
columns_with_lowNA = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond','MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']

#fill missing values for each column (using its own most frequent value)

object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])
#Number of null values in each feature

null_counts = numerical_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
print(len(numerical_columns_df['LotFrontage']))
print((numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']).median())

print(numerical_columns_df["LotFrontage"].median())

print(numerical_columns_df["MasVnrArea"].median())

print(numerical_columns_df["BsmtFinSF1"].median())

print(numerical_columns_df["BsmtFinSF2"].median())

print(numerical_columns_df["BsmtUnfSF"].median())

print(numerical_columns_df["TotalBsmtSF"].median())

print(numerical_columns_df["BsmtFullBath"].median())

print(numerical_columns_df["BsmtHalfBath"].median())

print(numerical_columns_df["GarageCars"].median())

print(numerical_columns_df["GarageArea"].median())

numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)

numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(0)

numerical_columns_df['MasVnrArea'] = numerical_columns_df['MasVnrArea'].fillna(0)

numerical_columns_df['BsmtFinSF1'] = numerical_columns_df['BsmtFinSF1'].fillna(368.5)

numerical_columns_df['BsmtFinSF2'] = numerical_columns_df['BsmtFinSF2'].fillna(0)

numerical_columns_df['BsmtUnfSF'] = numerical_columns_df['BsmtUnfSF'].fillna(467)

numerical_columns_df['TotalBsmtSF'] = numerical_columns_df['TotalBsmtSF'].fillna(989.5)

numerical_columns_df['BsmtFullBath'] = numerical_columns_df['BsmtFullBath'].fillna(0)

numerical_columns_df['BsmtHalfBath'] = numerical_columns_df['BsmtHalfBath'].fillna(0)

numerical_columns_df['GarageCars'] = numerical_columns_df['GarageCars'].fillna(2)

numerical_columns_df['GarageArea'] = numerical_columns_df['GarageArea'].fillna(480)
#Number of null values in each feature

null_counts = numerical_columns_df.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
numerical_columns_df.head()
numerical_columns_df.columns
numerical_columns_df.head()
object_columns_df['Utilities'].value_counts().plot(kind='bar',figsize=[10,3])

object_columns_df['Utilities'].value_counts() 
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
numerical_columns_df.columns
plt.scatter(numerical_columns_df.SalePrice,numerical_columns_df.Age_House)

plt.title('Price vs Age_House')
scaler = MinMaxScaler()



numerical_columns_df[['MSSubClass', 'LotFrontage', 'LotArea', 

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'TotalSA','OverallQual','GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'Age_House', 'TotalBsmtBath',

       'TotalBath' ]] = scaler.fit_transform(numerical_columns_df[['MSSubClass', 'LotFrontage', 'LotArea', 

       'OverallCond', 'YearBuilt', 'YearRemodAdd','TotalSA','OverallQual', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'Age_House', 'TotalBsmtBath',

       'TotalBath']])



numerical_columns_df.head()
object_columns_df.head()
#Using One hot encoder

object_columns_df = pd.get_dummies(object_columns_df, columns=object_columns_df.columns) 
object_columns_df.head()
df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)#

df_final.head()
df_final = df_final.drop(['Id',],axis=1)



df_train = df_final[df_final['train'] == 1]

df_train = df_train.drop(['train',],axis=1)





df_test = df_final[df_final['train'] == 0]

#df_test = df_test.drop(['SalePrice'],axis=1)

df_test = df_test.drop(['train','SalePrice'],axis=1)

df_test.head()

len(df_test)
target= df_train['SalePrice']

df_train = df_train.drop(['SalePrice'],axis=1)


print(len(df_train),len(target))
x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=0,num_parallel_tree=1,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=4, min_child_weight=2, n_estimators=2550,tree_method='exact',

             n_jobs=1, nthread=None, objective='reg:squarederror',

             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 

             silent=None, subsample=0.6, verbosity=1)





lgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=11000, 

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.38,

                                       )
y = target 

feature_names = df_train.columns

X = df_train[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

lgbm.fit(train_X, train_y)
xgb.fit(train_X, train_y)
predict1 = xgb.predict(x_test)

predict = lgbm.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

predict4 = lgbm.predict(df_test)

predict3 = xgb.predict(df_test)

predict_y =(predict3*.55+predict4*.45)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict_y    })

submission.to_csv('submission.csv', index=False)