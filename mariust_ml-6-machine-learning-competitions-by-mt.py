##### Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor




# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
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

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import numpy as np

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=42)

# fit rf_model_on_full_data on all data from the 
train_X = home_data.drop(['SalePrice', 'Id'], axis=1)
train_y = home_data.SalePrice
print(train_X.shape)

#dtypes_columns
#kinds make array with first letter of dtype columns
kinds = np.array([dt.kind for dt in train_X.dtypes])

all_cols = train_X.columns.values

#categorical columns
cat_cols = all_cols[kinds == 'O']

#numerical but categorical columns (num_cat columns)
year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']

#numerical columns without num_cat columns
num_cols = all_cols[kinds != 'O']
num_cols_2 = (train_X[num_cols].drop(year_cols, axis=1)).columns.values


#cat_pipeline
cat_pipeline = Pipeline([
    ('cat_si', SimpleImputer(strategy='most_frequent')),
    ('cat_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

#num_pipeline
num_pipeline = Pipeline([
    ('num_si', SimpleImputer(strategy='median')),
    ('num_ss', StandardScaler())
])          

#num_cat pipeline
year_pipeline = Pipeline([
    ('year_si', SimpleImputer(strategy='median')),
    ('year_kbins', KBinsDiscretizer(n_bins=5, encode='onehot-dense'))
])

#pipeline assembler         
ct = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols),
    ('num', num_pipeline, num_cols_2),
    ('year', year_pipeline, year_cols)
])

#transform train_X
train_X_prepared = ct.fit_transform(train_X)

#train model
rf_model_on_full_data.fit(train_X_prepared, train_y)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data.drop(['Id'], axis=1)
print(test_X.shape)

test_X_prepared = ct.transform(test_X)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X_prepared)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)