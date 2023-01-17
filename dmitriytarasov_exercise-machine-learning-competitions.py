# Code you have previously used to load data

import rpy2

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

#from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as mae

#from sklearn.metrics import mean_squared_log_error as msle



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# Dataset => Initial

iowa_data = pd.read_csv(iowa_file_path)

# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# Dataset => only numericals with imputation

y = iowa_data.SalePrice



train_y, val_y = train_test_split(y,random_state=0,train_size=0.75,test_size=0.25)



features = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',

       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',

       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold']





"""

X_data = iowa_data[features]

test_X = test_data[features]



train_X, val_X = train_test_split(X_data,random_state=0,train_size=0.75,test_size=0.25)



iowa_imputer = SimpleImputer()

train_X = iowa_imputer.fit_transform(train_X)

val_X = iowa_imputer.fit_transform(val_X)



model = XGBRegressor(n_estimators=1000,learning_rate=0.05)

model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X,val_y)],verbose=False)

predictions = model.predict(val_X)

print("Model1 Predictions : ")

print("MAE XGBR: {:,.0f}".format(mae(predictions, val_y)))

#print("RMSLE XGBR: {}".format(msle(predictions, val_y)))

"""



def one_hot_encoder(dataset):

    dataset_numerical = dataset.select_dtypes(exclude=['object'])

    imputer = SimpleImputer()

    dataset_imputed = imputer.fit_transform(dataset_numerical)

    dataset_imputed = pd.DataFrame(dataset_imputed)

    dataset_imputed.columns = dataset.select_dtypes(exclude=['object']).columns

    dataset_categorical = dataset.select_dtypes(include=['object'])

    dataset_imputed = dataset_imputed.join(dataset_categorical)

    dataset_one_hot = pd.get_dummies(dataset_imputed)

    return dataset_one_hot



def column_selector(dataset, features):

    dataset_object = dataset.select_dtypes(include=['object'])

    dataset_features = dataset[features]

    return dataset_object.join(dataset_features)



X_data = iowa_data.drop('SalePrice', axis=1)

X_data = column_selector(X_data, features)

X_data = one_hot_encoder(X_data)

test_X = test_data

test_X = column_selector(test_X, features)

test_X = one_hot_encoder(test_X)

train_X, val_X = train_test_split(X_data,random_state=0,train_size=0.75,test_size=0.25)



model = RandomForestRegressor(max_leaf_nodes=500,random_state=1,n_estimators=300)

model.fit(train_X, train_y)

predictions = model.predict(val_X)

print("Model Predictions : ")

print("MAE XGBR: {:,.0f}".format(mae(predictions, val_y)))

#print("RMSLE XGBR: {}".format(msle(predictions, val_y)))



# make predictions which we will submit. 

test_preds = model.predict(test_X)



print("Model Predictions : ")

print("MAE XGBR: {:,.0f}".format(mae(test_preds, val_y)))



# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)