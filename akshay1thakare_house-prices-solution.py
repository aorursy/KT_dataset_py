# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/test.csv')

test.head()
print('train has {} rows and {} features'.format(train.shape[0],train.shape[1]))

print('test has {} rows and {} features'.format(test.shape[0],test.shape[1]))
data = pd.concat([train.iloc[:,:-1],test],axis=0)

print('data has {} rows and {} features'.format(data.shape[0],data.shape[1]))
data.columns
categorical_features = data.select_dtypes(include='object')

numerical_features = data.select_dtypes(exclude='object')
numerical_features.describe()
categorical_features.describe()
data.isnull().sum().sort_values(ascending=False)[:34]
len(categorical_features.columns)
data = data.drop(columns=['Id','Street','PoolQC','Utilities'],axis=1)
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
features=['Electrical','Functional','KitchenQual','SaleType','Exterior2nd','Exterior1st','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']

for name in features:

    data[name].fillna('Other',inplace=True)
#data[features].isnull().sum()
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
zero=['GarageYrBlt','GarageArea','MasVnrArea','BsmtHalfBath','BsmtHalfBath','BsmtFullBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars']

for name in zero:

    data[name].fillna(0,inplace=True)
#check if our data has any nulls

data.isnull().sum().sum()
data.select_dtypes(include='object').columns
len(data.columns)
def dummies(d):

    dummies_df=pd.DataFrame()

    for name in d.select_dtypes(include='object').columns:

        dummies = pd.get_dummies(d[name], drop_first=False)

        dummies = dummies.add_prefix("{}_".format(name))

        dummies_df=pd.concat([dummies_df,dummies],axis=1)

    return dummies_df
dummies_data=dummies(data)

dummies_data.shape
object_features=['MSZoning','Alley', 'LotShape', 'LandContour',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive','Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']
data_non_obj=data.drop(columns=object_features,axis=1)

data_non_obj.shape
final_data=pd.concat([data_non_obj,dummies_data],axis=1)

final_data.shape
train_data=final_data.iloc[:1460,:]

test_data=final_data.iloc[1460:,:]

print(train_data.shape)

test_data.shape
X=train_data

y=train.loc[:,'SalePrice']
from sklearn.ensemble import RandomForestRegressor
rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)
test_X = test_data



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test.Id.values,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)