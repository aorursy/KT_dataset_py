#Loading some important libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # for splitting the data into training and validation sets
#Getting path for train and test data

train_path = '../input/train.csv'

# test_path = '../input/test.csv'



#Reading training and test data into DataFrames

data = pd.read_csv(train_path, index_col='Id')

# X_test_full = pd.read_csv(test_path, index_col='Id')
# Check first few rows of data

data.head()
# Separate target from predictors

y = data.SalePrice

X = data.drop(['SalePrice'], axis = 1)
# Print predictors

X.head()
# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train_full.shape
#Checking the null values in our training data

print("The total number of null values in our training data are: ", X_train_full.isnull().values.sum(), "\n")



#Printing the columns with null values

null_cols = X_train_full.isnull().sum()

null_cols = null_cols[null_cols>0]

print("Columns with their null values are: \n", null_cols)
# Selecting columns with maximum NaN values

nan_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']



# Dropping these columns from the training data

X_train_full = X_train_full.drop(nan_cols, axis=1)
# Selecting Categorical Columns with low cardinality - Cardinality means the number of unique values in a column

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 12 and X_train_full[cname].dtype == 'object']



# Printing number categorical_cols with low cardinality

print("The total number of categorical columns with < 12 unique values are: " , len(categorical_cols))



# Printing their names

print("\nThe column names are mentioned below:")

for col in categorical_cols:

    print(col)
# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtypes in ['int64', 'float64']]



# Printing the total number of numerical columns and their names

print("The total number of numerical columns are: ", len(numerical_cols))

print("\nThe column names are mentioned below: ")

for col in numerical_cols:

    print(col)
# Keep Selected columns

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()
X_train.head() 
# Importing some important libraries

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical columns - impute missing values

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical columns - impute missing values and apply One hot encoding

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle Preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numerical_cols),

    ('cat', categorical_transformer, categorical_cols)

])
# Importing XGBoost Model

from xgboost import XGBRegressor



model = XGBRegressor(n_estimators=1000, learnning_rate=0.05)
# Importing mean absolute error for using it as a metric to evaluate our model

from sklearn.metrics import mean_absolute_error



# Bundle Preprocessing and Modeling code in a pipeline

my_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])



# Preprocessing of training data, fit model

my_pipeline.fit(X_train, y_train)



# Processing of validation data, get predictions

predictions = my_pipeline.predict(X_valid)



# Calculate Mean Absolute Error

error = mean_absolute_error(y_valid, predictions)



# Print MAE

print("The mean absolute error is: ", error)
# Print first few predictions on validation set

print(predictions[0:5])
# Loading test set

X_test = pd.read_csv('../input/test.csv', index_col='Id')
# Predicting test set

test_predictions = my_pipeline.predict(X_test)



# Print first few predictions

print(test_predictions[0:5])
# Creating a Data Frame for submission file

submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': test_predictions})



#Creating a file name

filename = 'mysubmission.csv'



# Converting to csv

submission.to_csv(filename, index=False)



print("Saved File: " + filename)