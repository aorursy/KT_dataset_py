# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_train_file_path = '../input/train.csv'

iowa_test_file_path = '../input/test.csv'



train_data = pd.read_csv(iowa_train_file_path)

test_data = pd.read_csv(iowa_test_file_path)

#home_data = home_data.dropna(axis=0)

# Create target object and call it y_train

y_train = train_data.SalePrice

# Create X_train

X_train = train_data.drop(['SalePrice'], axis=1)

X_test = test_data



#find missing values

cols_with_missing = [col for col in X_train.columns

                            if X_train[col].isnull().any()]



reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_test = X_test.drop(cols_with_missing, axis=1)



cols_with_missing_test = [col for col in reduced_X_test.columns

                                 if reduced_X_test[col].isnull().any()]

reduced_X_train = reduced_X_train.drop(cols_with_missing_test, axis=1)

reduced_X_test = reduced_X_test.drop(cols_with_missing_test, axis=1)



reduced_numeric_X_train = reduced_X_train.select_dtypes(exclude=['object'])

reduced_numeric_X_test = reduced_X_test.select_dtypes(exclude=['object'])

# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=0)

rf_model.fit(reduced_numeric_X_train, y_train)
test_preds = rf_model.predict(reduced_numeric_X_test)
# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': X_test.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)