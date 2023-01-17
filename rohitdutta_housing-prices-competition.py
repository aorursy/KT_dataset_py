import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the data

import pandas as pd

sample_submission = pd.read_csv("../input/sample_submission.csv")

test_full = pd.read_csv("../input/test.csv")

train_full = pd.read_csv("../input/train.csv")



# Remove rows with missing target, separate target from predictors

train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_full.SalePrice

train_full.drop(['SalePrice'], axis=1, inplace=True)



# Separating numerical predictors

train_num = train_full.select_dtypes(exclude=['object'])

test_num = test_full.select_dtypes(exclude=['object'])



# Separating catagorical predictors

train_cat = train_full.select_dtypes(include=['object'])

test_cat = test_full.select_dtypes(include=['object'])
print(train_num.shape)

print(train_cat.shape)

print(test_num.shape)

print(test_cat.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
from sklearn.impute import SimpleImputer



#Imputation with median

final_imputer = SimpleImputer(strategy="median")



# Preprocessed training and validation features

final_train_num = pd.DataFrame(final_imputer.fit_transform(train_num))

final_test_num = pd.DataFrame(final_imputer.transform(test_num))



final_train_num.columns = train_num.columns

final_test_num.columns = test_num.columns
# train_cat.isnull().sum()

# test_cat.isnull().sum()



# Dropping columns with more than 600 missing values

drop_col = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]

train_cat.drop(drop_col, axis=1, inplace=True)

test_cat.drop(drop_col, axis=1, inplace=True)



'''# Dropping categorical predictors having missing values

cols_with_missing = [col for col in train_cat.columns if train_cat[col].isnull().any()]

train_cat.drop(cols_with_missing, axis=1, inplace=True)

test_cat.drop(cols_with_missing, axis=1, inplace=True)'''



print(train_cat.shape)

print(test_cat.shape)
# Imputing categorical columns with missing values by mode



train_cols_missing = [col for col in train_cat.columns if train_cat[col].isnull().any()]

for column in train_cols_missing:

    mode = train_cat[column].mode()

    train_cat[column].fillna(mode[0], inplace=True)



    

test_cols_missing = [col for col in test_cat.columns if test_cat[col].isnull().any()]

for column in test_cols_missing:

    mode = test_cat[column].mode()

    test_cat[column].fillna(mode[0], inplace=True)
object_cols = [col for col in train_cat.columns if train_cat[col].dtype == "object"]



# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if train_cat[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_cat[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test_cat[low_cardinality_cols]))





# One-hot encoding removed index; put it back

OH_cols_train.index = train_cat.index

OH_cols_test.index = test_cat.index
final_train = pd.concat([final_train_num, OH_cols_train], axis=1)

final_test = pd.concat([final_test_num, OH_cols_test], axis=1)
print(final_train.shape)

print(y.shape)

print(final_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_train, y, train_size=0.8, test_size=0.2, random_state=0)



# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(X_train, y_train)



# Get validation predictions and MAE

y_preds = model.predict(X_test)

print("MAE (Median Imputation):")

print(mean_absolute_error(y_test, y_preds))
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)],

             verbose=False)

predictions = my_model.predict(X_test)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))
#regressor = RandomForestRegressor(n_estimators=100, random_state=0)

#regressor.fit(final_train,y)



# Prediction

#preds_test = regressor.predict(final_test)



regressor = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

regressor.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)],

             verbose=False)



#Prediction

preds_test = regressor.predict(final_test)
# Save test predictions to file

output = pd.DataFrame({'Id': final_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)