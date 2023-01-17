# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
def get_encoded_nonnum_features(df, fs):

    new_df = pd.DataFrame()

    for feat in fs:

        new_df[feat] = df[feat].fillna('NA')

        new_df[feat] = LabelEncoder().fit_transform(new_df[feat])

    return new_df
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y

y = home_data.SalePrice



# Create X

features = ['LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',

            'FullBath', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual',

            'OverallCond', 'GarageYrBlt', 'ScreenPorch', 'WoodDeckSF']

X = home_data[features].fillna(0)



nonnum_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual']



X[nonnum_features] = get_encoded_nonnum_features(home_data, nonnum_features)

    

# le = LabelEncoder()

# X['LotShape'] = le.fit_transform(X['LotShape'])



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
def get_mae(n_estimators, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=1)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



def get_best_n_est(train_X, val_X, train_y, val_y):

    candidate_max_leaf_nodes = [i for i in range(50, 100)]



    best_cand, best_mae = -1, -1

    for candidate in candidate_max_leaf_nodes:

        curr_mae = get_mae(candidate, train_X, val_X, train_y, val_y)

        if best_mae == -1 or curr_mae < best_mae:

            best_cand = candidate

            best_mae = curr_mae

    return best_cand, best_mae
rf_model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

rf_model_on_full_data.fit(X, y)
test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features].fillna(0)

test_X[nonnum_features] = get_encoded_nonnum_features(test_data, nonnum_features)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)