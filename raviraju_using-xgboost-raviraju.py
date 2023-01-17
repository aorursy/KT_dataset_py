# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
home_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
# Create target object and call it y
y = home_data.SalePrice

X = home_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
features = X.columns

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(),y.as_matrix(), test_size=0.10)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
print(type(train_X))
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor()
my_model.fit(train_X, train_y)
preds = my_model.predict(test_X)
print(mean_absolute_error(preds, test_y))
# To improve accuracy, create a new Random Forest model which you will train on all training data
model_for_full_data = XGBRegressor(n_estimators=1100, learning_rate=0.04)

# fit rf_model_on_full_data on all data from the 
model_for_full_data.fit(X.as_matrix(), y.as_matrix(), verbose=False)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
# test_X = test_data[features]
competition_test = test_data[features]
type(test_data)
imputed_text_X = my_imputer.transform(competition_test)
type(imputed_text_X)
# make predictions which we will submit.
test_preds = model_for_full_data.predict(imputed_text_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)