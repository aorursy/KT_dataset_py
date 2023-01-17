# set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *

print("Setup Complete")
# load data

import pandas as pd

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full['SalePrice']

X_full.drop(['SalePrice'], axis=1, inplace=True)



# break off validation set from training data

from sklearn.model_selection import train_test_split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head()
from sklearn.pipeline import Pipeline



# preprocessing for numerical data

from sklearn.impute import SimpleImputer

numerical_transformer = SimpleImputer(strategy='constant')



# preprocessing for categorical data

from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# bundle preprocessing for numerical and categorical data

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# define model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)



# bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



from sklearn.metrics import mean_absolute_error

print('MAE:', mean_absolute_error(y_valid, preds))
# preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



# preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



# bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# define model

model = RandomForestRegressor(n_estimators=100,

                              min_samples_split=3,

                              random_state=0)



# check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

# step_1.a.hint()

# step_1.a.solution()
# bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)



# check your answer

step_1.b.check()
# line below will give you a hint

# step_1.b.hint()
# preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test)



# check your answer

step_2.check()
# lines below will give you a hint or solution code

# step_2.hint()

# step_2.solution()
# save test predictions to file

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)