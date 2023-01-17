!pip install pytorch-tabnet
import pandas as pd

import numpy  as np



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample     = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/solution.csv')
#===========================================================================

# select some features

#===========================================================================

features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 

            'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 

            '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 

            'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr',  'Fireplaces', 

            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

            'EnclosedPorch',  'PoolArea', 'YrSold']
X_train = train_data[features]

y_train = train_data["SalePrice"]



# make a validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)



X_test  = test_data[features]

y_true  = solution["SalePrice"]
X_train = X_train.apply(lambda x: x.fillna(x.mean()),axis=0)

X_valid = X_valid.apply(lambda x: x.fillna(x.mean()),axis=0)

X_test  = X_test.apply(lambda x: x.fillna(x.mean()),axis=0)
X_train = X_train.to_numpy()

y_train = y_train.to_numpy().reshape(-1, 1)

X_valid = X_valid.to_numpy()

y_valid = y_valid.to_numpy().reshape(-1, 1)

X_test  = X_test.to_numpy()
from pytorch_tabnet.tab_model import TabNetRegressor



regressor = TabNetRegressor()

regressor.fit(X_train=X_train, y_train=y_train,

              eval_set=[(X_valid, y_valid)],

              patience=500, max_epochs=3000)



predictions = regressor.predict(X_test)
from sklearn.metrics import mean_squared_log_error

RMSLE = np.sqrt( mean_squared_log_error(y_true, predictions) )

print("The score is %.5f" % RMSLE )
sample.iloc[:,1:] = predictions

sample.to_csv('submission.csv',index=False)