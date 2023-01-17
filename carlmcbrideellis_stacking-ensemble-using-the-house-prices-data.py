!pip install rgf_python
import pandas  as pd

import numpy   as np



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features

#===========================================================================

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 

        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 

        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 

        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 

        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

X_test        = test_data[features]



#===========================================================================

# imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

X_test       = X_test.fillna(X_test.mean())
from rgf.sklearn import RGFRegressor

import xgboost as xgb

from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import train_test_split



estimators =  [('xgb',xgb.XGBRegressor(n_estimators  = 750,learning_rate = 0.02, max_depth = 5)),

               ('cat',CatBoostRegressor(loss_function='RMSE', verbose=False)),

               ('RGF',RGFRegressor(max_leaf=500, algorithm="RGF_Sib", test_interval=100, loss="LS"))]



ensemble = StackingRegressor(estimators      =  estimators,

                             final_estimator =  RandomForestRegressor())



# Fit ensemble using cross-validation

X_tr, X_te, y_tr, y_te = train_test_split(X_train,y_train)

ensemble.fit(X_tr, y_tr).score(X_te, y_te)



# Prediction

predictions = ensemble.predict(X_test)
output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)