import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll use only numerical predictors

X = X_full.select_dtypes(exclude=['object'])

X_test = X_test_full.select_dtypes(exclude=['object'])



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns
print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
final_imputer = SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))

final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))



# Imputation removed column names; put them back

final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)

print("MAE (Your appraoch):")

print(mean_absolute_error(y_valid, preds_valid))