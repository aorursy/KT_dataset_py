import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read input data
input_data = pd.read_csv('../input/train.csv')

input_data.shape
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

target = input_data.SalePrice
train_data = input_data.drop(['SalePrice'], axis=1)
train_data = train_data.select_dtypes(exclude=['object'])
train_data = train_data.dropna(axis=1)

train_x, test_x, train_y, test_y = train_test_split(train_data, target, random_state=1)

xgb_model = XGBRegressor(n_learnings=1000)
xgb_model.fit(train_x, train_y, early_stopping_rounds=5, eval_set=[(test_x, test_y)], verbose=False)

predictions = xgb_model.predict(test_x)

err_rate = mean_absolute_error(predictions, test_y)
print(err_rate)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

max_nan_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage']
cols_to_drop = max_nan_cols + ['SalePrice', 'Id']
target = input_data.SalePrice

train_data = input_data.drop( cols_to_drop, axis=1)
# Find all the numeric columns
num_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']]
# Find all the categorical columns
dimensions = [col for col in train_data.columns if train_data[col].dtype == 'object' and train_data[col].nunique() < 10]

selected_cols = num_cols + dimensions
# Filter only the selected columns
selected_df = train_data[selected_cols]
one_hot_encoded_data = pd.get_dummies(selected_df)
one_hot_encoded_data.shape
my_imputer = SimpleImputer()

imputed_data = pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_data))
imputed_data.columns = one_hot_encoded_data.columns

train_x, test_x, train_y, test_y = train_test_split(imputed_data, target, random_state=1)

xgb_model = XGBRegressor(n_learning=1000, learning_rate=0.09)
xgb_model.fit(train_x, train_y, early_stopping_rounds=10, eval_set=[(test_x, test_y)], verbose=False)

predictions = xgb_model.predict(test_x)

err_rate = mean_absolute_error(predictions, test_y)
print(err_rate)
test_data = pd.read_csv('../input/test.csv')
selected_test_data = test_data[selected_cols]

# Get one-hot encoding for test data
one_hot_encoded_test_data = pd.get_dummies(selected_test_data)

# During One-Hot Encoding step, columns order in training and test data must have changed. Hence align using pd's align method
formatted_x, formatted_test_x = one_hot_encoded_data.align(one_hot_encoded_test_data, join='left', axis=1)
# Tips: Doing inner join selects only the common columns in training and test data. Hence we don't need to find the difference between
# two data set and delete columns that are not there in test

# # Find columns that present in training set but not test data
# # difference = set(one_hot_encoded_data.columns) - set(one_hot_encoded_test_data.columns)
# # one_hot_encoded_data = one_hot_encoded_data.drop(difference, axis=1)

imputer = SimpleImputer()
# Impute the training data
imputed_train_x = pd.DataFrame(imputer.fit_transform(formatted_x))
imputed_train_x.columns = formatted_x.columns

# Impute the test data
imputed_test_data = pd.DataFrame(imputer.transform(formatted_test_x))
imputed_test_data.columns = formatted_test_x.columns

# Split the train and test from imputed data so that we can use the test for eval_set in XGBRegressor
train_x, test_x, train_y, test_y = train_test_split(imputed_train_x, target, train_size=0.7, test_size=0.3, random_state=1)

# Create and train the model
xgb_model = XGBRegressor(n_learning=1000)
xgb_model.fit(train_x, train_y, early_stopping_rounds=5, eval_set=[(test_x, test_y)], verbose=False)

# Predict for the given test data
predictions = xgb_model.predict(imputed_test_data)

output_data = pd.DataFrame({ 'Id': test_data.Id, 'SalePrice': predictions})
output_data.to_csv('submission.csv', index=False)

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

cols_to_use = ['LotArea', 'YearBuilt']
x = imputed_train_x[cols_to_use]
y = target
model = GradientBoostingRegressor()
model.fit(x, y)

plot_partial_dependence(model,
                        features=[0, 1], 
                        feature_names=cols_to_use,
                        X=x, grid_resolution=10)

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

cols_to_use = ['OverallQual', 'YearRemodAdd', 'OverallCond']
x = imputed_train_x[cols_to_use]
y = target
model = GradientBoostingRegressor()
model.fit(x, y)

plot_partial_dependence(model,
                        features=[0, 1, 2], 
                        feature_names=cols_to_use,
                        X=x, grid_resolution=10)
partial_data = imputed_train_x[cols_to_use]
model = GradientBoostingRegressor()
model.fit(partial_data, target)
cols_to_use = ['GarageArea', 'GarageCars']

plot_partial_dependence(model,
                        features=[0, 1], 
                        feature_names=cols_to_use, X=partial_data, grid_resolution=10)