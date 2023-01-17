import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from fancyimpute import KNN

from sklearn.model_selection import train_test_split

import lightgbm as lgb



import os

print(os.listdir("../input"))
#Train data

train_df=pd.read_csv('../input/train.csv')

#Test data

test_df=pd.read_csv('../input/test.csv')

#submission file

sub_df=pd.read_csv('../input/sample_submission.csv')
#Displaying of train data

train_df.head()
#Displaying the test data

test_df.head()
#Feature names

train_df.columns
#Data types

train_df.dtypes.head()
#Shape of traain data

print(train_df.shape)

#Displaying summary of train data

train_df.describe()
#Shape of test data

print(test_df.shape)

#Displaying summary of test data

test_df.describe()
#Percentage of missing values present in train data

((train_df.isnull().sum())/1460 *100).iloc[0:10]
#Percentage of missing values present in test data

((test_df.isnull().sum())/1460 *100).iloc[0:10]
#Drop the unwanted variables

test_df.drop(['Id','PoolQC','Fence','MiscFeature','FireplaceQu','Alley'],axis=1).head()

train_df.drop(['Id','PoolQC','Fence','MiscFeature','FireplaceQu','Alley'],axis=1).head()
#train categorical attributes data

train_cat_attributes=train_df[['MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','MasVnrType','ExterQual','BsmtQual','BsmtExposure',

                        'BsmtFinType1','Heating','HeatingQC','CentralAir','Electrical','Functional','GarageType','GarageFinish','KitchenQual','GarageQual','PavedDrive','SaleType','SaleCondition']]



#train numeric attributes data

train_num_attributes=train_df[['LotFrontage','LotArea','OverallQual','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',

                         'FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','SalePrice']]
#imputing the missing values by using Simple Imputer

imp = SimpleImputer(strategy="most_frequent")

impute_cat=imp.fit_transform(train_cat_attributes)

print(impute_cat)

train_cat_attributes=pd.DataFrame(impute_cat,columns=train_cat_attributes.columns)

train_cat_attributes.head()
#imputing the missing values by using Simple Imputer

imp = SimpleImputer(strategy="mean")

impute_num=imp.fit_transform(train_num_attributes)

print(impute_num)

train_num_attributes=pd.DataFrame(impute_num,columns=train_num_attributes.columns)

train_num_attributes.head()
#Scatter plot for different zones(MSZoning)

fig=plt.figure(figsize=(10,5))

#fig,ax=plt.subplots(1,2)

sns.scatterplot(x=train_cat_attributes['MSZoning'],y=train_num_attributes['SalePrice'])



#Scatter plot for street

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['Street'],y=train_num_attributes['SalePrice'])
#Scatter plot for Land contour

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['LandContour'],y=train_num_attributes['SalePrice'])



#Scatter plot for Land config

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['LotConfig'],y=train_num_attributes['SalePrice'])
#Scatter plot for Land frontage

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_num_attributes['LotFrontage'],y=train_num_attributes['SalePrice'])

#Scatter plot for lot area

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_num_attributes['LotArea'],y=train_num_attributes['SalePrice'])
#Scatter plot fortype of House Style

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['HouseStyle'],y=train_num_attributes['SalePrice'])



#Scatter plot for type of Roof Style

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['RoofStyle'],y=train_num_attributes['SalePrice'])
#Scatter plot for type of Heating

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['Heating'],y=train_num_attributes['SalePrice'])



#Scatter plot for Heating Quality

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['HeatingQC'],y=train_num_attributes['SalePrice'])
#Scatter plot for Electrical

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['Electrical'],y=train_num_attributes['SalePrice'])



#Scatter plot for Central Air

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['CentralAir'],y=train_num_attributes['SalePrice'])
#Scatter plot for Garage type

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['GarageType'],y=train_num_attributes['SalePrice'])

#Scatter plot for Kitchen quality

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['MasVnrType'],y=train_num_attributes['SalePrice'])
#Scatter plot for Garage cars

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_num_attributes['GarageCars'],y=train_num_attributes['SalePrice'])

#Scatter plot for Garage Area

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_num_attributes['GarageArea'],y=train_num_attributes['SalePrice'])
#Scatter plot for Sale type

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['SaleType'],y=train_num_attributes['SalePrice'])

#Scatter plot for Sale condition

plt.figure(figsize=(10,5))

sns.scatterplot(x=train_cat_attributes['SaleCondition'],y=train_num_attributes['SalePrice'])
#Encoding train categorical attributes

train_cat_attributes=pd.get_dummies(train_cat_attributes,columns=train_cat_attributes.columns)

train_cat_attributes.head()
#Categorical test attributes

test_cat_attributes=test_df[['MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','MasVnrType','ExterQual','BsmtQual','BsmtExposure',

                        'BsmtFinType1','Heating','HeatingQC','CentralAir','Electrical','Functional','GarageType','GarageFinish','KitchenQual','GarageQual','PavedDrive','SaleType','SaleCondition']]



#Numeric test attributes

test_num_attributes=test_df[['LotFrontage','LotArea','OverallQual','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',

                         'FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch','PoolArea','MiscVal']]
#imputing the missing values by using Simple Imputer

imp = SimpleImputer(strategy="most_frequent")

impute_cat=imp.fit_transform(test_cat_attributes)

print(impute_cat)

test_cat_attributes=pd.DataFrame(impute_cat,columns=test_cat_attributes.columns)

test_cat_attributes.head()
#imputing the missing values by using Simple Imputer

imp = SimpleImputer(strategy="mean")

impute_num=imp.fit_transform(test_num_attributes)

print(impute_num)

test_num_attributes=pd.DataFrame(impute_num,columns=test_num_attributes.columns)

test_num_attributes.head()
#Encoding test categorical attributes

test_cat_attributes=pd.get_dummies(test_cat_attributes,columns=test_cat_attributes.columns)

test_cat_attributes.head()
#Merging categorical and numeric train attributes

train_attributes=pd.merge(train_cat_attributes,train_num_attributes,left_index=True,right_index=True)

train_attributes.shape
#Merging categorical and numeric test attributes

test_attributes=pd.merge(test_cat_attributes,test_num_attributes,left_index=True,right_index=True)

test_attributes.head()

test_attributes.shape
#Correlations in train attributes

attributes=train_attributes.columns.values

correlations=train_attributes[attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

train_correlations=correlations[correlations['level_0']!=correlations['level_1']]

print(train_correlations.tail())
#drop the unwanted train attributes

train_attributes=train_attributes.drop(['CentralAir_Y','Street_Grvl','SaleCondition_Partial','RoofStyle_Gable','LotShape_IR1', 'LandSlope_Gtl','ExterQual_TA','GarageArea','Neighborhood_Somerst','Electrical_FuseA',

                                       'PavedDrive_N','RoofMatl_Tar&Grv','GarageQual_Fa','TotRmsAbvGrd','KitchenQual_TA','GarageType_Detchd','TotalBsmtSF','HouseStyle_2Story','MSZoning_RL','BsmtQual_TA','MasVnrType_None'],axis=1)

print(train_attributes.shape)

train_attributes.head()
#Correlations in test attributes

attributes=test_attributes.columns.values

correlations=test_attributes[attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

test_correlations=correlations[correlations['level_0']!=correlations['level_1']]

print(test_correlations.tail())
#drop the unwanted test attributes

test_attributes=test_attributes.drop(['Street_Grvl','CentralAir_N','SaleCondition_Partial','LandSlope_Gtl','RoofStyle_Gable','LotShape_IR1','GarageQual_Fa','GarageArea','ExterQual_TA','PavedDrive_N','Neighborhood_Somerst',

                     'Electrical_FuseA','RoofMatl_Tar&Grv','Heating_GasW','KitchenQual_Gd','MasVnrType_None','HouseStyle_2Story','MSZoning_RL','GarageType_Detchd'],axis=1)

print(test_attributes.shape)

test_attributes.head()
#Split the training dataset

X=train_attributes.drop(['SalePrice'],axis=1)

y=train_attributes.SalePrice

X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.1)

print(X_train.shape,y_train.shape)

print(X_valid.shape,y_valid.shape)
#Setting dataset for lightgbm model

lgb_train=lgb.Dataset(X_train,label=y_train)

lgb_valid=lgb.Dataset(X_valid,label=y_valid)
#Choosing the parameters

params={'boosting_type': 'gbdt', 

          'max_depth' : 20,

          'objective': 'regression',

          'boost_from_average':False, 

          'nthread': 8,

          'num_leaves': 120,

          'learning_rate': 0.05,

          'min_data_in_leaf':30,

          'bagging_fraction':0.8,

          'max_bin': 500,  

          'subsample_for_bin': 100,

          'metric' : 'rmse',

          }
%%time

#training the model

num_round=2000

lgbm= lgb.train(params,lgb_train,num_round,valid_sets=[lgb_train,lgb_valid],verbose_eval=100,early_stopping_rounds = 1500)

lgbm
#predict the model

lgbm_predict=lgbm.predict(test_attributes,random_state=42,num_iteration=lgbm.best_iteration)

lgbm_predict
#plot importance features

lgb.plot_importance(lgbm,max_num_features=50,importance_type="split",figsize=(15,10))
df1=pd.DataFrame(lgbm_predict,columns=['SalePrice'])

df2=pd.DataFrame(test_df['Id'],columns=['Id'])



submission=pd.merge(df2,df1,left_index=True,right_index=True)

submission.set_index('Id')

submission.to_csv('House Price Predictions')