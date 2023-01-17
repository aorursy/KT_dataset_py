# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder, OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

import matplotlib.pyplot as plt

%matplotlib inline

from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)

import category_encoders as ce

from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from pandas_profiling import ProfileReport

import math
def save_file (predictions):

    """Save submission file."""

    # Save test predictions to file

    output = pd.DataFrame({'Id': sample_submission_file.Id,

                       'SalePrice': predictions})

    output.to_csv('submission.csv', index=False)

    print ("Submission file is saved")



def calculate_root_mean_squared_log_error(y_true, y_pred):

    """Calculate root mean squared error of log(y_true) and log(y_pred)"""

    if len(y_pred)!=len(y_true): return 'error_mismatch'

    y_pred_new = [math.log(x+1) for x in y_pred]

    y_true_new = [math.log(x+1) for x in y_true]

    return mean_squared_error(y_true_new, y_pred_new, squared=False)
# Read the data

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

X = train_data.copy()



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice', 'Utilities'], axis=1, inplace=True)

X_test.drop(['Utilities'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



sample_submission_file = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")



print('Data is OK')
# Select object columns

categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]



# Select numeric columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]



# Number of missing values in each column of training data

missing_val_count_by_column_train = (X_train.isnull().sum())

print("Number of missing values in each column:")

print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
# Number of missing values in numerical columns

missing_val_count_by_column_numeric = (X_train[numerical_cols].isnull().sum())

print("Number of missing values in numerical columns:")

print(missing_val_count_by_column_numeric[missing_val_count_by_column_numeric > 0])
# Imputation lists



# imputation to null values of these numerical columns need to be 'constant'

constant_num_cols = ['GarageYrBlt', 'MasVnrArea']



# imputation to null values of these numerical columns need to be 'mean'

mean_num_cols = list(set(numerical_cols).difference(set(constant_num_cols)))



# imputation to null values of these categorical columns need to be 'constant'

constant_categorical_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                             'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu','GarageType',

                             'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',

                             'MiscFeature']



# imputation to null values of these categorical columns need to be 'most_frequent'

mf_categorical_cols = list(set(categorical_cols).difference(set(constant_categorical_cols)))



my_cols = constant_num_cols + mean_num_cols + constant_categorical_cols + mf_categorical_cols
# Define transformers

# Preprocessing for numerical data



numerical_transformer_m = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())])



numerical_transformer_c = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),

    ('scaler', StandardScaler())])





# Preprocessing for categorical data for most frequent

categorical_transformer_mf = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])



# Preprocessing for categorical data for constant

categorical_transformer_c = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])





# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num_mean', numerical_transformer_m, mean_num_cols),

        ('num_constant', numerical_transformer_c, constant_num_cols),

        ('cat_mf', categorical_transformer_mf, mf_categorical_cols),

        ('cat_c', categorical_transformer_c, constant_categorical_cols)

    ])
# Define Model

model = XGBRegressor(learning_rate = 0.01,

                            n_estimators=2500,

                            max_depth=5,

                            min_child_weight=1,

                            gamma=0,

                            subsample=0.8,

                            colsample_bytree=0.8,

                            reg_alpha = 0,

                            reg_lambda = 1,

                            random_state=0)
# Preprocessing of validation data

X_valid_eval = preprocessor.fit(X_train, y_train).transform (X_valid)
# Display the number of remaining columns after transformation 

print("We have", X_valid_eval.shape[1], "features after transformation")
# Define XGBRegressor fitting parameters for the pipeline

fit_params = {"model__early_stopping_rounds": 50,

              "model__eval_set": [(X_valid_eval, y_valid)],

              "model__verbose": False,

              "model__eval_metric" : "rmsle"}
# Create and Evaluate the Pipeline

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train, **fit_params)



# Get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = calculate_root_mean_squared_log_error(y_valid,preds)



print("Score: {}".format(score))

# Preprocessing of training data, fit model 

X_cv = X[my_cols].copy()

X_sub = X_test[my_cols].copy()
# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X_cv, y,

                              cv=5,

                              scoring=make_scorer(calculate_root_mean_squared_log_error,

                                                  greater_is_better=False))



print("Score:\n", scores)

print("Score mean: {}".format(scores.mean()))

print("Score std: {}".format(scores.std()))
# Fit model

my_pipeline.fit(X_cv, y)



# Get predictions

preds = my_pipeline.predict(X_sub)
# Use predefined utility function

save_file(preds)