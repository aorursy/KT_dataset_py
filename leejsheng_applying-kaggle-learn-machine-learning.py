import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as mae



from xgboost import XGBRegressor



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Path of files

train_data_path = '../input/train.csv'

test_data_path = '../input/test.csv'



train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)



#print(train_data.info())

#print(test_data.info())
# extract features, X and label, y

features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

            'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 

            'GarageArea', 'PoolArea']



# X with specified features

# X = train_data[features]



# X with ALL numeric-only features

X = train_data.drop(['Id','SalePrice'], axis=1).select_dtypes(exclude=['object'])

y = train_data.SalePrice



#print(X.info())
# seperating data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.info())



mean_imputer = SimpleImputer() # default to mean

X_train = mean_imputer.fit_transform(X_train) # fit the statistics to training set then impute it

X_val = mean_imputer.transform(X_val) # impute validation set with statistics of training set

#print(pd.DataFrame(X_train).info())
def get_mae (max_leaf_nodes, max_depth, X_train, X_val, y_train, y_val):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, random_state=42)

    model.fit(X_train, y_train)

    error = mae(y_val, model.predict(X_val))

    

    return error
# find optimal settings

candidate_max_leaf_nodes = [50, 100, 150, 200, 250, 300]

candidate_max_depth = [8, 16, 32, 64, 128]



best_error = float('Inf')

best_max_leaf_nodes = 0

best_max_depth = 0



for max_depth in candidate_max_depth:

    for max_leaf_nodes in candidate_max_leaf_nodes:

        error = get_mae(max_leaf_nodes, max_depth, X_train, X_val, y_train, y_val)

        #print('Mx Lf Nds: %d \t Mx Dpth: %d \t mae: %f' %(max_leaf_nodes,max_depth,error))

        

        if error < best_error:

            best_error = error

            best_max_leaf_nodes = max_leaf_nodes

            best_max_depth = max_depth

            

print("Best: Mx-Lf-Nds: %d, Mx-Dpth: %d \t @mae: %f" %(best_max_leaf_nodes,best_max_depth,best_error))

# build model with best settings

dt_model = DecisionTreeRegressor(max_leaf_nodes=best_max_leaf_nodes, max_depth=best_max_depth, random_state=42)

dt_model.fit(X_train, y_train)

dt_mae = mae(y_val, dt_model.predict(X_val))

print("Validation MAE for Decision Tree Regressor: {:,.0f}".format(dt_mae))
rf_model = RandomForestRegressor(random_state=100)

rf_model.fit(X_train, y_train)

rf_mae = mae(y_val, rf_model.predict(X_val))

print("Validation MAE for Random Forest Regressor: {:,.0f}".format(rf_mae))
xgb_model = XGBRegressor()

xgb_model.fit(X_train, y_train, verbose=False)

xgb_mae = mae(y_val, xgb_model.predict(X_val))

print("Validation MAE for XGBoost Regressor: {:,.0f}".format(xgb_mae))
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.10004)

xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)], verbose=False) #make verbose=True to find the best n_estimator value

xgb_mae = mae(y_val, xgb_model.predict(X_val))

print("Validation MAE for XGBoost Regressor: {:,.0f}".format(xgb_mae))

# this example was 190
# let's process our training features

mean_imputer = SimpleImputer() # default to mean

X_train_full = mean_imputer.fit_transform(X)

#print(pd.DataFrame(X_train_full).info())



# apply same preprocessing on test data that we did on training data

X_test = test_data.drop(['Id'], axis=1).select_dtypes(exclude=['object'])

X_test = mean_imputer.transform(X_test)

#print(pd.DataFrame(X_test).info())
# train the model with chosen values

xgb_model_full = XGBRegressor(n_estimators=190, learning_rate=0.10004)

xgb_model_full.fit(X_train_full, y, verbose=False)
test_predictions = xgb_model_full.predict(X_test)

output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_predictions})

output.to_csv('submission.csv', index=False)