import numpy as np              # for working with numerical data in Python

import pandas as pd             # for data manipulation and analysis

import matplotlib.pyplot as plt # loading allies as plt for plotting 

from sklearn import preprocessing  # importing preprossing to clean and process our data

from sklearn.model_selection import train_test_split # loading to split our data into training and testing

from lightgbm import LGBMRegressor # loading gradient boosting regression for implementation

from xgboost import XGBRegressor  # loading XG boosting algorithm for accuracy comparision with lgbm

import sklearn.metrics as metrics # loading algebric matrics processor

import math # to do some descriptive statistics
test = pd.read_csv("../input/datafiles/test.csv")

train = pd.read_csv("../input/datafiles/train.csv")

#Creating a copy of the train and test datasets

Copy_test  = test.copy()

Copy_train  = train.copy()
Copy_train.head()
Copy_test.head()
Copy_train['train']  = 1

Copy_test['train']  = 0

df = pd.concat([Copy_train, Copy_test], axis=0,sort=False)
NAN = [(c, df[c].isna().mean()*100) for c in df]                    # In this code we are extracting out NAN values by using .isna() function and calculating mean 

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])      # In this code, we are preparing a datasframe with the NAN values.
NAN = NAN[NAN.percentage > 50] # extracting columns in which null values are more than 50% of whole dataset.

NAN.sort_values("percentage", ascending=False) # as we have used pd.DataFrame...we can use sort_values module in that...hence extracting out sorted NAN values
#Drop PoolQC, MiscFeature, Alley and Fence features

df = df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
object_columns_df = df.select_dtypes(include=['object'])

numerical_columns_df =df.select_dtypes(exclude=['object'])
object_columns_df.dtypes
numerical_columns_df.dtypes
null_counts = object_columns_df.isnull().sum()  # Number of null values in each feature

print("Number of null values in each column:\n{}".format(null_counts)) # using print formatting feature ... printing those values
columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']

object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')
columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']



#fill missing values for each column (using its own most frequent value)

object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])
# Calculating Number of null values in each feature



null_counts = numerical_columns_df.isnull().sum()



print("Number of null values in each column:\n{}".format(null_counts))
print((numerical_columns_df['YrSold']- numerical_columns_df['YearBuilt']).median())

print(numerical_columns_df["LotFrontage"].median())
numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)

numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)
numerical_columns_df= numerical_columns_df.fillna(0)
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
df_final = df_final.drop(['Id',],axis=1)



df_train = df_final[df_final['train'] == 1]

df_train = df_train.drop(['train',],axis=1)



df_test = df_final[df_final['train'] == 0]

df_test = df_test.drop(['SalePrice'],axis=1)

df_test = df_test.drop(['train',],axis=1)

target= df_train['SalePrice']

df_train = df_train.drop(['SalePrice'],axis=1)
x_train,x_test,y_train,y_test = train_test_split(df_train,target,test_size=0.33,random_state=0)
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

predict_y = ( predict3*0.45 + predict4 * 0.55)



import matplotlib

print(predict4,predict_y)

plt.plot(predict4,predict3)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict_y

    })

submission.to_csv('submission.csv', index=False)