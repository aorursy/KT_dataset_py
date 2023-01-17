import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import random

import numpy as np  # linear algebra

import pandas as pd  #

from datetime import datetime



from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



from mlxtend.regressor import StackingCVRegressor



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

random.seed(1)


train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.info()

# Converting MSZoning to numerical

train_data.MSZoning.value_counts(dropna = False).plot.bar()

trans_dict = {'RL':0,'RM':1,'FV':2,'RH':3,'C (all)':4}

train_data.MSZoning = train_data.MSZoning.replace(trans_dict)

test_data.MSZoning = test_data.MSZoning.replace(trans_dict)
#converting LotShape to numerical

train_data.LotShape.value_counts(dropna = False).plot.bar()

trans_dict = {'Reg':0,'IR1':1,'IR2':2,'IR3':3}

train_data.LotShape = train_data.LotShape.replace(trans_dict)

test_data.LotShape = test_data.LotShape.replace(trans_dict)
#converting Street to numerical

train_data.Street.value_counts(dropna = False).plot.bar()

trans_dict = {'Pave':0,'Grvl':1}

train_data.Street = train_data.Street.replace(trans_dict)

test_data.Street = test_data.Street.replace(trans_dict)

#Handling Alley , as it has lot of NaN we will drop it.

train_data.Alley.value_counts(dropna = False).plot.bar()

train_data.drop('Alley',inplace = True,axis = 1) 

test_data.drop('Alley',inplace = True,axis = 1) 
# Handling LandContour

train_data.LandContour.value_counts(dropna = False).plot.bar()

trans_dict = {'Lvl':0,'Bnk':1,'HLS':2,'Low':3}

train_data.LandContour = train_data.LandContour.replace(trans_dict)

test_data.LandContour = test_data.LandContour.replace(trans_dict)
#Handling Utilities

train_data.Utilities.value_counts(dropna = False).plot.bar()

# as it has a lot of skewness towards the AllPub, we can drop it, it is not that important for the model.

train_data.drop('Utilities',inplace = True,axis = 1)

test_data.drop('Utilities',inplace = True,axis = 1)
#Handling LotConfig

train_data.LotConfig.value_counts(dropna = False).plot.bar()

trans_dict = {'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4}

train_data.LotConfig = train_data.LotConfig.replace(trans_dict)

test_data.LotConfig = test_data.LotConfig.replace(trans_dict)
# Handling LandSlope

train_data.LandSlope.value_counts(dropna = False).plot.bar()

trans_dict = {'Gtl':0,'Mod':1,'Sev':2}

train_data.LandSlope = train_data.LandSlope.replace(trans_dict)

test_data.LandSlope = test_data.LandSlope.replace(trans_dict)
#Handling Neighborhood

train_data.Neighborhood.value_counts(dropna = False).plot.bar()

le = LabelEncoder()

train_data.Neighborhood = le.fit_transform(train_data.Neighborhood)

test_data.Neighborhood = le.transform(test_data.Neighborhood)

#Handling Condition1

train_data.Condition1.value_counts(dropna = False).plot.bar()

train_data.Condition1 = le.fit_transform(train_data.Condition1)

test_data.Condition1 = le.transform(test_data.Condition1)

#Handling Condition2

train_data.Condition2.value_counts(dropna = False).plot.bar()

# very skewed therefore we can drop it

train_data.drop('Condition2',axis = 1,inplace = True)

test_data.drop('Condition2',axis = 1,inplace = True)

#Handling BldgType 

train_data.BldgType.value_counts(dropna = False).plot.bar()

train_data.BldgType = le.fit_transform(train_data.BldgType)

test_data.BldgType = le.transform(test_data.BldgType)

#Handling HouseStyle

print(train_data.HouseStyle.value_counts(dropna = False))

train_data.HouseStyle.value_counts(dropna = False).plot.bar()

train_data.HouseStyle = le.fit_transform(train_data.HouseStyle)

test_data.HouseStyle = le.transform(test_data.HouseStyle)

#Handling RoofStyle

print(train_data.RoofStyle.value_counts(dropna = False))

train_data.RoofStyle.value_counts(dropna = False).plot.bar()

train_data.RoofStyle = le.fit_transform(train_data.RoofStyle)

test_data.RoofStyle = le.transform(test_data.RoofStyle)

#Handling RoofMatl

print(train_data.RoofMatl.value_counts(dropna = False))

train_data.RoofMatl.value_counts(dropna = False).plot.bar()

print("Skewed towards a single feature :",train_data.RoofMatl.value_counts()[0]/len(train_data))

#very skewed therefore dropping it.

train_data.drop('RoofMatl',axis = 1,inplace = True)

test_data.drop('RoofMatl',axis = 1,inplace = True)

#Handling Exterior1st

print(train_data.Exterior1st.value_counts(dropna = False))

print(test_data.Exterior1st.value_counts(dropna = False))

train_data.Exterior1st.value_counts(dropna = False).plot.bar()

test_data.Exterior1st.fillna('VinylSd',inplace = True)

# there is a lot of values which are not in train but in test therefore drop it.

le = le.fit(train_data.Exterior1st.unique().tolist()+test_data.Exterior1st.unique().tolist())

train_data.Exterior1st = le.transform(train_data.Exterior1st)

test_data.Exterior1st = le.transform(test_data.Exterior1st)

#Handling Exterior2nd

print(train_data.Exterior2nd.value_counts(dropna = False))

print(test_data.Exterior2nd.value_counts(dropna = False))

train_data.Exterior2nd.value_counts(dropna = False).plot.bar()

test_data.Exterior2nd.fillna('VinylSd',inplace = True)  #filling the nan or null values in test data

le = le.fit(train_data.Exterior2nd.unique().tolist()+test_data.Exterior2nd.unique().tolist())

train_data.Exterior2nd = le.transform(train_data.Exterior2nd)

test_data.Exterior2nd = le.transform(test_data.Exterior2nd)

#Handling MasVnrType

print(train_data.MasVnrType.value_counts(dropna = False))

print(test_data.MasVnrType.value_counts(dropna = False))

train_data.MasVnrType.value_counts(dropna = False).plot.bar()

test_data.MasVnrType.fillna('None',inplace = True)

train_data.MasVnrType.fillna('None',inplace = True)

le = le.fit(train_data.MasVnrType.unique().tolist()+test_data.MasVnrType.unique().tolist())

train_data.MasVnrType = le.transform(train_data.MasVnrType)

test_data.MasVnrType = le.transform(test_data.MasVnrType)



#Handling ExterQual

print(train_data.ExterQual.value_counts(dropna = False))

print(test_data.ExterQual.value_counts(dropna = False))

train_data.ExterQual.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.ExterQual.unique().tolist())

train_data.ExterQual = le.transform(train_data.ExterQual)

test_data.ExterQual = le.transform(test_data.ExterQual)

#Handling ExterCond

print(train_data.ExterCond.value_counts(dropna = False))

print(test_data.ExterCond.value_counts(dropna = False))

train_data.ExterCond.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.ExterCond.unique().tolist())

train_data.ExterCond = le.transform(train_data.ExterCond)

test_data.ExterCond = le.transform(test_data.ExterCond)

#Handling Foundation

print(train_data.Foundation.value_counts(dropna = False))

print(test_data.Foundation.value_counts(dropna = False))

train_data.Foundation.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.Foundation.unique().tolist())

train_data.Foundation = le.transform(train_data.Foundation)

test_data.Foundation = le.transform(test_data.Foundation)



#Handling BsmtQual

print(train_data.BsmtQual.value_counts(dropna = False))

print(test_data.BsmtQual.value_counts(dropna = False))

train_data.BsmtQual.value_counts(dropna = False).plot.bar()

test_data.BsmtQual.fillna('TA',inplace = True)

train_data.BsmtQual.fillna('TA',inplace = True)

le = le.fit(train_data.BsmtQual.unique().tolist())

train_data.BsmtQual = le.transform(train_data.BsmtQual)

test_data.BsmtQual = le.transform(test_data.BsmtQual)





#Handling BsmtCond

print(train_data.BsmtCond.value_counts(dropna = False))

print(test_data.BsmtCond.value_counts(dropna = False))

train_data.BsmtCond.value_counts(dropna = False).plot.bar()

test_data.BsmtCond.fillna('TA',inplace = True)

train_data.BsmtCond.fillna('TA',inplace = True)

le = le.fit(train_data.BsmtCond.unique().tolist())

train_data.BsmtCond = le.transform(train_data.BsmtCond)

test_data.BsmtCond = le.transform(test_data.BsmtCond)



#Handling BsmtExposure

print(train_data.BsmtExposure.value_counts(dropna = False))

print(test_data.BsmtExposure.value_counts(dropna = False))

train_data.BsmtExposure.value_counts(dropna = False).plot.bar()

test_data.BsmtExposure.fillna('No',inplace = True)

train_data.BsmtExposure.fillna('No',inplace = True)

le = le.fit(train_data.BsmtExposure.unique().tolist())

train_data.BsmtExposure = le.transform(train_data.BsmtExposure)

test_data.BsmtExposure = le.transform(test_data.BsmtExposure)





#Handling BsmtFinType1

print(train_data.BsmtFinType1.value_counts(dropna = False))

print(test_data.BsmtFinType1.value_counts(dropna = False))

train_data.BsmtFinType1.value_counts(dropna = False).plot.bar()

test_data.BsmtFinType1.fillna('GLQ',inplace = True)

train_data.BsmtFinType1.fillna('Unf',inplace = True)

le = le.fit(train_data.BsmtFinType1.unique().tolist())

train_data.BsmtFinType1 = le.transform(train_data.BsmtFinType1)

test_data.BsmtFinType1 = le.transform(test_data.BsmtFinType1)



#Handling BsmtFinType2

print(train_data.BsmtFinType2.value_counts(dropna = False))

print(test_data.BsmtFinType2.value_counts(dropna = False))

train_data.BsmtFinType2.value_counts(dropna = False).plot.bar()

test_data.BsmtFinType2.fillna('Unf',inplace = True)

train_data.BsmtFinType2.fillna('Unf',inplace = True)

le = le.fit(train_data.BsmtFinType2.unique().tolist())

train_data.BsmtFinType2 = le.transform(train_data.BsmtFinType2)

test_data.BsmtFinType2 = le.transform(test_data.BsmtFinType2)





#Handling Heating

print(train_data.Heating.value_counts(dropna = False))

print(test_data.Heating.value_counts(dropna = False))

train_data.Heating.value_counts(dropna = False).plot.bar()

#very skewed therefore drop it.

test_data.drop('Heating',axis = 1,inplace = True)

train_data.drop('Heating',axis = 1,inplace = True)





#Handling HeatingQC

print(train_data.HeatingQC.value_counts(dropna = False))

print(test_data.HeatingQC.value_counts(dropna = False))

train_data.HeatingQC.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.HeatingQC.unique().tolist())

train_data.HeatingQC = le.transform(train_data.HeatingQC)

test_data.HeatingQC = le.transform(test_data.HeatingQC)





#Handling CentralAir

print(train_data.CentralAir.value_counts(dropna = False))

print(test_data.CentralAir.value_counts(dropna = False))

train_data.CentralAir.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.CentralAir.unique().tolist())

train_data.CentralAir = le.transform(train_data.CentralAir)

test_data.CentralAir = le.transform(test_data.CentralAir)





#Handling Electrical

print(train_data.Electrical.value_counts(dropna = False))

print(test_data.Electrical.value_counts(dropna = False))

train_data.Electrical.value_counts(dropna = False).plot.bar()

#very skewed therefore we can drop it.

train_data.drop('Electrical',axis = 1,inplace = True)

test_data.drop('Electrical',axis = 1,inplace = True)





#Handling KitchenQual

print(train_data.KitchenQual.value_counts(dropna = False))

print(test_data.KitchenQual.value_counts(dropna = False))

train_data.KitchenQual.value_counts(dropna = False).plot.bar()

test_data.KitchenQual.fillna('TA',inplace = True)

le = le.fit(train_data.KitchenQual.unique().tolist())

train_data.KitchenQual = le.transform(train_data.KitchenQual)

test_data.KitchenQual = le.transform(test_data.KitchenQual)





#Handling Functional

print(train_data.Functional.value_counts(dropna = False))

print(test_data.Functional.value_counts(dropna = False))

train_data.Functional.value_counts(dropna = False).plot.bar()

#very skewed therefore we can drop it.

train_data.drop('Functional',axis = 1,inplace = True)

test_data.drop('Functional',axis = 1,inplace = True)





#Handling FireplaceQu

print(train_data.FireplaceQu.value_counts(dropna = False))

print(test_data.FireplaceQu.value_counts(dropna = False))

train_data.FireplaceQu.value_counts(dropna = False).plot.bar()

test_data.FireplaceQu.fillna('None',inplace = True)

train_data.FireplaceQu.fillna('None',inplace = True)

le = le.fit(train_data.FireplaceQu.unique().tolist())

train_data.FireplaceQu = le.transform(train_data.FireplaceQu)

test_data.FireplaceQu = le.transform(test_data.FireplaceQu)





#Handling GarageType

print(train_data.GarageType.value_counts(dropna = False))

print(test_data.GarageType.value_counts(dropna = False))

train_data.GarageType.value_counts(dropna = False).plot.bar()

test_data.GarageType.fillna('Attchd',inplace = True)

train_data.GarageType.fillna('Attchd',inplace = True)

le = le.fit(train_data.GarageType.unique().tolist())

train_data.GarageType = le.transform(train_data.GarageType)

test_data.GarageType = le.transform(test_data.GarageType)





#Handling GarageFinish

print(train_data.GarageFinish.value_counts(dropna = False))

print(test_data.GarageFinish.value_counts(dropna = False))

train_data.GarageFinish.value_counts(dropna = False).plot.bar()

test_data.GarageFinish.fillna('None',inplace = True) # filling nan or null values with None

train_data.GarageFinish.fillna('None',inplace = True) # filling nan or null values with None

le = le.fit(train_data.GarageFinish.unique().tolist())

train_data.GarageFinish = le.transform(train_data.GarageFinish)

test_data.GarageFinish = le.transform(test_data.GarageFinish)





#Handling GarageQual

print(train_data.GarageQual.value_counts(dropna = False))

print(test_data.GarageQual.value_counts(dropna = False))

train_data.GarageQual.value_counts(dropna = False).plot.bar()

#very skewed we can drop it.

train_data.drop('GarageQual',axis = 1,inplace = True)

test_data.drop('GarageQual',axis = 1,inplace = True)



#Handling GarageCond

print(train_data.GarageCond.value_counts(dropna = False))

print(test_data.GarageCond.value_counts(dropna = False))

train_data.GarageCond.value_counts(dropna = False).plot.bar()

#very skewed we can drop it.

train_data.drop('GarageCond',axis = 1,inplace = True)

test_data.drop('GarageCond',axis = 1,inplace = True)

#Handling PavedDrive

print(train_data.PavedDrive.value_counts(dropna = False))

print(test_data.PavedDrive.value_counts(dropna = False))

train_data.PavedDrive.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.PavedDrive.unique().tolist())

train_data.PavedDrive = le.transform(train_data.PavedDrive)

test_data.PavedDrive = le.transform(test_data.PavedDrive)

#Handling PoolQC

print(train_data.PoolQC.value_counts(dropna = False))

print(test_data.PoolQC.value_counts(dropna = False))

train_data.PoolQC.value_counts(dropna = False).plot.bar()

#too much NaN values we can drop it.

train_data.drop('PoolQC',axis = 1,inplace = True)

test_data.drop('PoolQC',axis = 1,inplace = True)

#Handling Fence

print(train_data.Fence.value_counts(dropna = False))

print(test_data.Fence.value_counts(dropna = False))

train_data.Fence.value_counts(dropna = False).plot.bar()

#too much NaN values we can drop it.

train_data.drop('Fence',axis = 1,inplace = True)

test_data.drop('Fence',axis = 1,inplace = True)

#Handling MiscFeature

print(train_data.MiscFeature.value_counts(dropna = False))

print(test_data.MiscFeature.value_counts(dropna = False))

train_data.MiscFeature.value_counts(dropna = False).plot.bar()

#too much NaN values we can drop it.

train_data.drop('MiscFeature',axis = 1,inplace = True)

test_data.drop('MiscFeature',axis = 1,inplace = True)

#Handling SaleType

print(train_data.SaleType.value_counts(dropna = False))

print(test_data.SaleType.value_counts(dropna = False))

train_data.SaleType.value_counts(dropna = False).plot.bar()

#very skewed we can drop it.

train_data.drop('SaleType',axis = 1,inplace = True)

test_data.drop('SaleType',axis = 1,inplace = True)

#Handling SaleCondition

print(train_data.SaleCondition.value_counts(dropna = False))

print(test_data.SaleCondition.value_counts(dropna = False))

train_data.SaleCondition.value_counts(dropna = False).plot.bar()

le = le.fit(train_data.SaleCondition.unique().tolist())

train_data.SaleCondition = le.transform(train_data.SaleCondition)

test_data.SaleCondition = le.transform(test_data.SaleCondition)

#Let's get the shape of our data to know number of features and samples

train_data.shape
#Let's get our X_train and y_train

X_train = train_data.drop(['SalePrice'],axis = 1)

y_train = train_data['SalePrice']

X_train.fillna(method = 'ffill',inplace = True)
#Let's prepare our training and testing data

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2)
#preparing data for lightgbm

lgb_train = lgb.Dataset(X_train,y_train)

lgb_test = lgb.Dataset(X_test,y_test)
#Let's set parameters for our model

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'num_leaves': 31,

    'learning_rate': 0.01,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}

#training our lightgbm model

model = lgb.train(params,lgb_train,num_boost_round=10000,valid_sets=lgb_test,early_stopping_rounds=100)
test_data.head()
test_data.fillna(method = 'ffill',inplace = True)

pred = model.predict(test_data)

sample = pd.read_csv('../input/sample_submission.csv')

sample['SalePrice'] = pred

sample.to_csv('sub.csv',index = False)

sample.head()