import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

train_data = pd.read_csv('../input/train.csv', header=0)

test_data = pd.read_csv('../input/test.csv', header=0)
train_data.head()
categorical_features = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',

                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',

                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',

                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',

                      'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',

                      'MiscFeature','SaleType','SaleCondition']

predictors = [col for col in train_data.columns if col not in ['SalePrice','Id']]

train_data.describe()
data = pd.concat([train_data,test_data])

data.head()
data.shape
non_categorical_cols= [col for col in data.columns if col not in categorical_features and col not in ['Id'] ]

numeric_feats = data[non_categorical_cols].dtypes[data.dtypes != "object"].index

data[numeric_feats] = np.log1p(data[numeric_feats])
data.head()
SalePrice = data.SalePrice

data.dropna(axis=1,inplace=True)

data['SalePrice'] = SalePrice
data.head()
categorical_features = list(set(categorical_features).intersection(set(data.columns)))

data = pd.get_dummies(data,columns =categorical_features)
train_set = data.loc[data.SalePrice.notna()]

test_set = data.loc[data.SalePrice.isna()]
train_X = train_set[train_set.columns[train_set.columns.values!='SalePrice']]

train_y = train_set.SalePrice

train_X.drop('Id',axis=1,inplace=True)

test_X = test_set[test_set.columns[test_set.columns.values!='SalePrice']]
model = xgboost.XGBRegressor(colsample_bytree=0.4,gamma=0,learning_rate=0.07,max_depth=3,min_child_weight=1.5,

                             n_estimators=10000,reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,seed=42)
model.fit(train_X,train_y)
submission=pd.DataFrame(columns=['Id','SalePrice'])

submission['Id']=test_set['Id']

test_X.drop('Id',axis=1,inplace=True)

submission['SalePrice']=model.predict(test_X)

submission['SalePrice']=submission['SalePrice'].apply(lambda x: np.expm1(x))

submission.head()
submission.to_csv('xgb.csv',index=False)