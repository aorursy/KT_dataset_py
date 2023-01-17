# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read data to variables
home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv(test_data_path)

# Create target object and call it y
y = home_data.SalePrice

X = home_data.select_dtypes(include=[np.number]).drop('SalePrice', axis=1)
test_X = test_data.select_dtypes(include=[np.number])

train_X, val_X, train_y, val_y = train_test_split(X.as_matrix(), y.as_matrix(), random_state=1)

my_imputer = SimpleImputer()

test_X = my_imputer.fit_transform(test_X)
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.fit_transform(val_X)

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(train_X, train_y, verbose=False
              , early_stopping_rounds=5, eval_set=[(val_X, val_y)] )

val_preds = xgb_model.predict(val_X)
val_mae = mean_absolute_error(val_preds, val_y)

print("Validation MAE for XGBoost model: {:,.0f}".format(val_mae))
# predict by using XGBoost model
test_preds = xgb_model.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
