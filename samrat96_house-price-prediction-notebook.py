import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import math

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test = pd.read_csv("../input/home-data-for-ml-course/test.csv")

train = pd.read_csv("../input/home-data-for-ml-course/train.csv")
new_test  = test.copy()

new_train  = train.copy()
new_train.head()
new_test.head()
sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(new_train['SalePrice'], color="r");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
corr = new_train.corr()

plt.subplots(figsize=(15,12))

sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
new_train['train']  = 1

new_test['train']  = 0

new_data=pd.concat([new_train, new_test], axis=0,sort=False)
def percentage_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x



missing = percentage_missing(new_data)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percentage of missing data')

df_miss[0:10]
new_data = new_data.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
object_columns=new_data.select_dtypes(include=['object'])

numerical_columns=new_data.select_dtypes(exclude=['object'])
#Categorical Features

object_columns.dtypes
#Numerical columns

numerical_columns.dtypes
#Null values in each categorical feature

null_values = object_columns.isnull().sum()

print("Null values for each categorical feature:\n{}".format(null_values))
columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']

object_columns[columns_None]= object_columns[columns_None].fillna('None')
columns_with_low_null_values = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']



object_columns[columns_with_low_null_values] = object_columns[columns_with_low_null_values].fillna(object_columns.mode().iloc[0])

object_columns.describe()
#Number of null values in each numeric feature

null_values = numerical_columns.isnull().sum()

print("Null values of each numeric values:\n{}".format(null_values))
print((numerical_columns['YrSold']-numerical_columns['YearBuilt']).median())

print(numerical_columns["LotFrontage"].median())
numerical_columns['GarageYrBlt'] = numerical_columns['GarageYrBlt'].fillna(numerical_columns['YrSold']-35)

numerical_columns['LotFrontage'] = numerical_columns['LotFrontage'].fillna(68)
numerical_columns= numerical_columns.fillna(0)
numerical_columns.describe()
object_columns['Street'].value_counts().plot(kind='bar',figsize=[12,8])

object_columns['Street'].value_counts() 
object_columns['Heating'].value_counts().plot(kind='bar',figsize=[12,8])

object_columns['Heating'].value_counts()
object_columns['RoofMatl'].value_counts().plot(kind='bar',figsize=[12,8])

object_columns['RoofMatl'].value_counts() 
object_columns['Utilities'].value_counts().plot(kind='bar',figsize=[12,8])

object_columns['Utilities'].value_counts() 
object_columns['Condition2'].value_counts().plot(kind='bar',figsize=[12,8])

object_columns['Condition2'].value_counts()
object_columns = object_columns.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)
numerical_columns['Age_of_House']= (numerical_columns['YrSold']-numerical_columns['YearBuilt'])

numerical_columns['Age_of_House'].describe()
negative_age_of_house=numerical_columns[numerical_columns['Age_of_House'] < 0]

negative_age_of_house
numerical_columns.loc[numerical_columns['YrSold'] < numerical_columns['YearBuilt'],'YrSold' ] = 2010
numerical_columns['Age_of_House']= (numerical_columns['YrSold']-numerical_columns['YearBuilt'])

numerical_columns['Age_of_House'].describe()
numerical_columns['TotalBsmtBath'] = numerical_columns['BsmtFullBath'] + numerical_columns['BsmtFullBath']*0.5

numerical_columns['TotalBath'] = numerical_columns['FullBath'] + numerical_columns['HalfBath']*0.5 

numerical_columns['TotalSA']=numerical_columns['TotalBsmtSF'] + numerical_columns['1stFlrSF'] + numerical_columns['2ndFlrSF']
numerical_columns.head()
object_columns.head()
mapping  = {'TA':1,'Gd':3, 'Fa':2,'Ex':5,'Po':7,'None':0,'Y':1,'N':1,'Reg':3,'IR1':2,

            'IR2':1,'IR3':0,"None" : 0,"No" : 5, "Mn" : 4, "Av": 3,

            "Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 2, "ALQ" : 3, "GLQ" : 6 }

Paveddrive_mapping =   {"N" : 1, "P" : 2, "Y" : 3}

object_columns['ExterQual'] = object_columns['ExterQual'].map(mapping)

object_columns['ExterCond'] = object_columns['ExterCond'].map(mapping)

object_columns['BsmtCond'] = object_columns['BsmtCond'].map(mapping)

object_columns['BsmtQual'] = object_columns['BsmtQual'].map(mapping)

object_columns['HeatingQC'] = object_columns['HeatingQC'].map(mapping)

object_columns['KitchenQual'] = object_columns['KitchenQual'].map(mapping)

object_columns['FireplaceQu'] = object_columns['FireplaceQu'].map(mapping)

object_columns['GarageQual'] = object_columns['GarageQual'].map(mapping)

object_columns['GarageCond'] = object_columns['GarageCond'].map(mapping)

object_columns['CentralAir'] = object_columns['CentralAir'].map(mapping)

object_columns['LotShape'] = object_columns['LotShape'].map(mapping)

object_columns['BsmtExposure'] = object_columns['BsmtExposure'].map(mapping)

object_columns['BsmtFinType1'] = object_columns['BsmtFinType1'].map(mapping)

object_columns['BsmtFinType2'] = object_columns['BsmtFinType2'].map(mapping)

object_columns['PavedDrive'] = object_columns['PavedDrive'].map(Paveddrive_mapping)
remaining_columns = object_columns.select_dtypes(include=['object'])



object_columns = pd.get_dummies(object_columns, columns=remaining_columns.columns) 
data = pd.concat([object_columns, numerical_columns], axis=1,sort=False)

data.head()
data = data.drop(['Id',],axis=1)

train_data = data[data['train'] == 1]

train_data = train_data.drop(['train',],axis=1)



test_data = data[data['train'] == 0]

test_data = test_data.drop(['SalePrice'],axis=1)

test_data = test_data.drop(['train',],axis=1)
y= train_data['SalePrice']

x = train_data.drop(['SalePrice'],axis=1)
#TRAIN AND TEST SPLITTING

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=0.6, gamma=0,importance_type='gain',

                  learning_rate=0.02, max_delta_step=0,max_depth=4, min_child_weight=1.5, n_estimators=2000,n_jobs=1, nthread=None,

                  objective='reg:linear',reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, silent=None,subsample=0.8, verbosity=1)





lgbm = LGBMRegressor(objective='regression',num_leaves=4,learning_rate=0.01, n_estimators=12000, max_bin=200, bagging_fraction=0.75,

                                       bagging_freq=5, bagging_seed=7,feature_fraction=0.4)  
xgb.fit(x_train, y_train)

lgbm.fit(x_train, y_train,eval_metric='rmse')
predict_xgb = xgb.predict(x_test)

predict_lgbm = lgbm.predict(x_test)
import sklearn.metrics as metrics

print('RMSE test XGB = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict_xgb))))

print('RMSE test LGBM = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict_lgbm))))
#Model fitting on whole dataset

xgb.fit(x, y)

lgbm.fit(x,y,eval_metric='rmse')
new_predict_xgb = xgb.predict(test_data)

new_predict_lgbm = lgbm.predict(test_data)

predict = ( new_predict_xgb*0.45 + new_predict_lgbm * 0.55)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predict})

submission.to_csv('submission.csv', index=False)