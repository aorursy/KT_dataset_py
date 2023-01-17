!pip install xgbtune
import pandas  as pd

import xgboost as xgb



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
from xgbtune import tune_xgb_model

params = {'eval_metric': 'rmsle'}

params, round_count = tune_xgb_model(params, X_train, y_train)
#===========================================================================

# now use the parameters from XGBTune

#===========================================================================

regressor=xgb.XGBRegressor(**params)



regressor.fit(X_train, y_train)



#===========================================================================

# use the fit to predict the prices for the test data

#===========================================================================

predictions = regressor.predict(X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)