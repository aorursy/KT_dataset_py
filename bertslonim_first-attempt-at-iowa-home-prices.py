#setup notebook
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Load data
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# check that the data loaded correctly
print(home_data.shape)
print(home_data.columns)
#get a feel for the data
#home_data.info()
home_data.head()
home_target = home_data.SalePrice
home_predictors = home_data.drop(['SalePrice'], axis=1)

# For the sake of keeping things simple, use only numeric predictors; drop non-numerics. 
home_numeric_predictors = home_predictors.select_dtypes(exclude=['object'])
#home_numeric_predictors.info()

# Split the numeric predictors into test-train groups
X_train, X_test, y_train, y_test = train_test_split(home_numeric_predictors, 
                                                    home_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# function to assess model mean_absolute_error

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# impute missing data and determine mean_absolute_error
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# add columns to keep track of which values are missing in the original data
# these columns are simply boolean vectors indicating whether the corresponding value is missing

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

# list comprehension to identify any columns with missing values
cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())

# load booleans to indicate whether the corresponding value is imputed
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

print(imputed_X_train_plus.columns)
print(imputed_X_train_plus.GarageYrBlt_was_missing.head(10))
# Now that we know where the missing values are located perform Imputation to generate imputed values
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

