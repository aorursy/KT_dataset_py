# Code you have previously used to load data

import pandas as pd

# from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2



from sklearn.impute import SimpleImputer
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice



features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



X = home_data[features]



my_imputer = SimpleImputer()

i_X = pd.DataFrame(my_imputer.fit_transform(X))

i_X.columns = X.columns

X = i_X



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# X.sample()

# import missingno

# missingno.matrix(X)
# rksh_model = RandomForestRegressor(random_state=1)

# rksh_model = RandomForestRegressor(n_estimators=80, random_state=1)



# from xgboost import XGBRegressor

# rksh_model = XGBRegressor(n_estimators=125)

# rksh_model.fit(train_X, train_y)

# pred_y = rksh_model.predict(val_X)

# mean_absolute_error(val_y, pred_y) # 15053
# rf_model_on_full_data = RandomForestRegressor(n_estimators=80, random_state=1)

rf_model_on_full_data = XGBRegressor(n_estimators=212)



rf_model_on_full_data.fit(X,y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# # read test data file using pandas

test_data = pd.read_csv(test_data_path)



test_X = test_data[features]



i_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))

i_test_X.columns = test_X.columns

test_X = i_test_X



test_preds = rf_model_on_full_data.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)