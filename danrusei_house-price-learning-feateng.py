import pandas as pd
#Random Forest Regressor - A random forest is a meta estimator that fits a number of classifying decision trees 
# on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
from sklearn.ensemble import RandomForestRegressor
# Mean absolute error regression loss
from sklearn.metrics import mean_absolute_error
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# Imputation transformer for completing missing values.
from sklearn.impute import SimpleImputer

# path to the file to read
iowa_file_path = '../input/train.csv'
# read into a PD DataFrame
home_data = pd.read_csv(iowa_file_path)

# y
home_target = home_data.SalePrice
# X
home_predictors = home_data.drop(['SalePrice'], axis=1)
# For simplicity we try only with numeric predictors
home_num_predictors = home_predictors.select_dtypes(exclude=['object'])
# Understand data, check missing values @ count

missing_val_count_by_column = (home_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
X_train, X_test, y_train, y_test = train_test_split(home_num_predictors, home_target, train_size=0.7, test_size=0.3, random_state=0) 

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
cols_with_missing = [ col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
X_columns = imputed_X_train_plus.columns
y_columns =imputed_X_test_plus.columns

my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(imputed_X_train_plus))
imputed_X_test_plus = pd.DataFrame(my_imputer.transform(imputed_X_test_plus))
imputed_X_train_plus.columns = X_columns
imputed_X_test_plus.columns = y_columns

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

#path to file for prediction
test_data_path = '../input/test.csv'

#read test data using pandas
test_data = pd.read_csv(test_data_path)
test_data = test_data.select_dtypes(exclude=['object'])

test_data_plus = test_data.copy()

for col in cols_with_missing:
    test_data_plus[col + '_was_missing'] = test_data_plus[col].isnull()
test_columns = test_data_plus.columns
test_data_plus = pd.DataFrame(my_imputer.fit_transform(test_data_plus))
test_data_plus.columns = test_columns

model = RandomForestRegressor()
model.fit(imputed_X_train_plus, y_train)
test_preds = model.predict(test_data_plus)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)