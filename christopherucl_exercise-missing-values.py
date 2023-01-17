# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex2 import *

print("Setup Complete")
#X_full.head
#X_full_copy = pd.read_csv('../input/train.csv', index_col='Id')
#X_full_copy['SalePrice'].isna().sum()
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
#X_train.head()
#print(missing_val_count_by_column)

#type(missing_val_count_by_column)
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#X_train.dtypes
# Fill in the line below: How many rows are in the training data?

num_rows = 1168



# Fill in the line below: How many columns in the training data

# have missing values?

num_cols_with_missing = 3



# Fill in the line below: How many missing entries are contained in 

# all of the training data?

tot_missing = 276



# Check your answers

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
#step_1.b.hint()
#step_1.b.solution()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Fill in the line below: get names of columns with missing values

cols_with_missing = [i for i in X_train.columns

                     if X_train[i].isnull().any()] # Your code here



# Fill in the lines below: drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis =1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
print("MAE (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# from sklearn.impute import SimpleImputer

# my_imputer = SimpleImputer() # Your code here

# imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
#X_train.columns # gives u column names
#imputed_X_train.head()
from sklearn.impute import SimpleImputer



# Fill in the lines below: imputation

my_imputer = SimpleImputer() # Your code here

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution()
print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
#step_3.b.hint()
#step_3.b.solution()
#X_train.columns
#X_train.isna().sum()[X_train.isna().sum() >0]
#X_train.LotFrontage.describe()

#X_train.shape

# X_train_no_NA = X_train.MasVnrArea.dropna()

# X_train_no_NA.describe()

#X_train_no_NA.mean()
X_train_copy = X_train.copy()

X_train_copy.LotFrontage.fillna(69, inplace=True)
X_train_copy.MasVnrArea.fillna(0, inplace=True)


X_train_copy.GarageYrBlt.fillna(1979, inplace=True)
#X_valid_copy.isna().sum()
X_valid_copy = X_valid.copy()

X_valid_copy.LotFrontage.fillna(69, inplace=True)

X_valid_copy.MasVnrArea.fillna(0, inplace=True)

X_valid_copy.GarageYrBlt.fillna(1979, inplace=True)
# Preprocessed training and validation features

final_X_train = X_train_copy

final_X_valid = X_valid_copy



# Check your answers

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution()
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)

print("MAE (Your appraoch):")

print(mean_absolute_error(y_valid, preds_valid))
X_test_copy = X_test.copy()

X_test_copy.LotFrontage.fillna(69, inplace=True)

X_test_copy.MasVnrArea.fillna(0, inplace=True)

X_test_copy.GarageYrBlt.fillna(1979, inplace=True)



X_test_copy.BsmtFinSF1.fillna(350, inplace=True)

X_test_copy.BsmtFinSF2.fillna(0, inplace=True)

X_test_copy.BsmtUnfSF.fillna(460, inplace=True)

X_test_copy.TotalBsmtSF.fillna(988, inplace=True)

X_test_copy.BsmtFullBath.fillna(0, inplace=True)

X_test_copy.BsmtHalfBath.fillna(0, inplace=True)

X_test_copy.GarageCars.fillna(2, inplace=True)

X_test_copy.GarageArea.fillna(480, inplace=True)



#X_test_copy.isna().sum()
#X_train.LotFrontage.describe()

#X_train.shape

# X_train_no_NA = X_test_copy.GarageArea.dropna()

# X_train_no_NA.describe()

#X_train_no_NA.mean()
# Number of missing values in each column of training data

# missing_val_count_by_column2 = (X_test_copy.isnull().sum())

# print(missing_val_count_by_column2[missing_val_count_by_column2 > 0])
#X_test_copy.shape

#X_test_copy.isna().sum()
# Fill in the line below: preprocess test data  X_test_copy

final_X_test = X_test_copy



# Fill in the line below: get test predictions

preds_test = model.predict(final_X_test)



step_4.b.check()
# Lines below will give you a hint or solution code

#step_4.b.hint()

#step_4.b.solution()
preds_test
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)