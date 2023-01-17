
features=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterCond', 'Foundation',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt','GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition']
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'

data = pd.read_csv(iowa_file_path,na_values="?")


features=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterCond', 'Foundation',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt','GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition']

X=data[features]

X['MasVnrType']=X.MasVnrType.fillna("NULL")

cleanup={"MasVnrType" : {"NULL": 0 ,"None":0 ,"BrkFace" :1 ,"Stone":2 ,"BrkCmn" :3} ,
         "Heating" : {'GasA':1, 'GasW':0 ,'Grav':0 , 'Wall': 0 ,'OthW':0 ,'Floor':0}
        }

X.replace(cleanup,inplace=True)

le=LabelEncoder()

for col_name in X.columns:
    if X[col_name].dtype=="float64" or X[col_name].dtype=="int64":
        if X[col_name].isna().any() or X[col_name].isnull().any():
            X[col_name]=X[col_name].fillna(0)

for col_name in X.columns:
  if X[col_name].dtype=="object":
    X[col_name]=X[col_name].fillna("NULL")
    X[col_name]=le.fit_transform(X[col_name])

y=data["SalePrice"]
X.corrwith(y)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'

data = pd.read_csv(iowa_file_path,na_values="?")


features=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterCond', 'Foundation',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt','GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition']

X=data[features]

X['MasVnrType']=X.MasVnrType.fillna("NULL")

cleanup={"MasVnrType" : {"NULL": 0 ,"None":0 ,"BrkFace" :1 ,"Stone":2 ,"BrkCmn" :3} ,
         "Heating" : {'GasA':1, 'GasW':0 ,'Grav':0 , 'Wall': 0 ,'OthW':0 ,'Floor':0}
        }

X.replace(cleanup,inplace=True)

le=LabelEncoder()

for col_name in X.columns:
    if X[col_name].dtype=="float64" or X[col_name].dtype=="int64":
        if X[col_name].isna().any() or X[col_name].isnull().any():
            X[col_name]=X[col_name].fillna(0)

for col_name in X.columns:
  if X[col_name].dtype=="object":
    X[col_name]=X[col_name].fillna("NULL")
    X[col_name]=le.fit_transform(X[col_name])

y=data["SalePrice"]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=0,n_estimators=1000,verbose=0,
                                              criterion='mae',max_features='sqrt',oob_score=True,n_jobs=4,)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(train_X,train_y)
# path to file you will use for predictions
test_data_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
    

features=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterCond', 'Foundation',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt','GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition']

test_X = test_data[features]


le=LabelEncoder()
for col_name in test_X.columns:
  if test_X[col_name].dtype=="object":
    test_X[col_name]=test_X[col_name].fillna("NULL")
    test_X[col_name]=le.fit_transform(test_X[col_name])
    
for col in test_X.columns:
    if test_X[col].isna().any() or test_X[col].isnull().any():
        test_X[col]=test_X[col].fillna(0)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
output.head()