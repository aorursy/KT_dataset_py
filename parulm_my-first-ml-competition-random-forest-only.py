# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR



# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

iowa_target = home_data.SalePrice
iowa_predictors = home_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

iowa_numeric_predictors.head()
#Load test data and do one hot encoding on both datasets before imputations
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

#one hot encoding
ohe_training = pd.get_dummies(iowa_predictors)
ohe_test = pd.get_dummies(test_data)
cur_train, cur_test = ohe_training.align(ohe_test, join='inner', axis=1)
from sklearn.impute import SimpleImputer

#new_data = iowa_numeric_predictors.copy()
#new_data = iowa_predictors.copy()
new_data = cur_train.copy()

# make new columns indicating what will be imputed
cols_with_missing = list(col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    print(col)
    new_data[col + '_was_missing'] = new_data[col].isnull()

my_imputer = SimpleImputer()
new_data_imputed = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data_imputed.columns = new_data.columns
#new_data.columns
#tester code block. delete later
cur_test.columns
# path to file you will use for predictions
#test_data_path = '../input/test.csv'

# read test data file using pandas
#test_data = pd.read_csv(test_data_path)

#new_test = test_data[iowa_numeric_predictors.columns]
new_test = cur_test

for col in cols_with_missing:
    print(col)
    new_test[col + '_was_missing'] = new_test[col].isnull()

#new_test.head()
# Imputation
#my_imputer = SimpleImputer()
new_test_imput = pd.DataFrame(my_imputer.fit_transform(new_test))
new_test_imput.columns = new_test.columns
test_X = new_test_imput

#One hot encoding while aligning the train and test datas
#one_hot_encoded_training_predictors = pd.get_dummies(new_data_imputed)
#one_hot_encoded_test_predictors = pd.get_dummies(test_X)
#final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
#                                                                    join='left', 
#                                                                    axis=1)
final_train = new_data_imputed
final_test = test_X
#list(final_train.columns)
X = new_data_imputed
y = iowa_target
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
#list(train_X.columns)
#One hot encoding just for testing purposes
#one_hot_encoded_training_predictors = pd.get_dummies(train_X)
#one_hot_encoded_test_predictors = pd.get_dummies(val_X)
#train_X, val_X = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
#                                                                    join='left', 
#                                                                    axis=1)
from xgboost import XGBRegressor

#note to self: n_estimators=140 was earlier optimized
iowa_model_xgb = XGBRegressor(n_estimators=250)
#iowa_model_xgb.fit(train_X, train_y, early_stopping_rounds=20, 
#                   eval_set=[(val_X, val_y)], verbose=True)
iowa_model_xgb.fit(train_X, train_y)

predictions_xgb = iowa_model_xgb.predict(val_X)
xgb_val_mae = mean_absolute_error(predictions_xgb, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
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
rf_model = RandomForestRegressor(random_state=1, n_estimators=100)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

#using support vector regression
#clf = SVR()
#clf.fit(train_X, train_y)
#svm_pred = clf.predict(val_X)
#svm_mae = mean_absolute_error(svm_pred, val_y)
#print(svm_mae)
# To improve accuracy, create a new Random Forest model which you will train on all training data
#rf_model_on_full_data = RandomForestRegressor(random_state=1, n_estimators=100)
rf_model_on_full_data = XGBRegressor(n_estimators=250)

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(final_train, y)

#Making predictions
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
