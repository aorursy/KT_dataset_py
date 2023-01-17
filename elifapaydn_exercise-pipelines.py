# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex4 import *

print("Setup Complete")
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv',index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

X_full=X_full[my_cols]

X_train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.model_selection import GridSearchCV



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))
# finding optimum max_leaf_nodes

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=max_leaf_nodes, random_state=0)

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

    my_pipeline.fit(train_X, train_y)

    preds = my_pipeline.predict(val_X)

    mae = mean_absolute_error(val_y, preds)

    return(mae)



for max_leaf_nodes in [5, 50, 100, 500, 2000, 5000]:

    my_mae = get_mae(max_leaf_nodes, X_train, X_valid,y_train, y_valid)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant') # Your code here



# Preprocessing for categorical data

categorical_transformer = Pipeline([('impute', SimpleImputer(strategy='constant')), ('encode', OneHotEncoder(handle_unknown='ignore'))]) # Your code here



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=500, max_leaf_nodes=2000, random_state=0) # Your code here



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

r2 = r2_score(y_valid, preds)

print('MAE:{}, coeff of determination: {}'.format(score,r2))



# Check your answer

step_1.b.check()
param_grid={'model__n_estimators': [100,200,500],

            "model__max_features": ["auto", "sqrt", "log2"],

            "model__min_samples_split" : [2,4,8,10,50],

            "model__max_leaf_nodes":[100,500,1000,2000,10000]

            }

# "model__max_leaf_nodes":[100,500,1000,2000]

#"model__max_depth":np.arange(1,12,2),

# "model__bootstrap":[True, False]



rf=RandomForestRegressor(random_state=0)

pipe=Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', rf)])

gs=GridSearchCV(pipe, param_grid=param_grid, scoring={'MAE':'neg_mean_absolute_error','coeff of det':'r2'},refit='MAE')

gs.fit(X_train, y_train)

print('tuned parameters:' ,gs.best_params_)

print('best MAE:' ,gs.best_score_)
gs.cv_results_
cvresults=gs.cv_results_

idx=np.nonzero(cvresults['rank_test_MAE']==1)[0]

r2score=cvresults['mean_test_coeff of det'][idx[0]]

r2score
#new model with tuned parameters

model=RandomForestRegressor(max_features='sqrt', max_leaf_nodes=1000, min_samples_split=2, n_estimators=500,random_state=0)

my_pipeline=Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])

my_pipeline.fit(X_train,y_train)

y_pred=my_pipeline.predict(X_valid)



# Evaluate the tuned model

score = mean_absolute_error(y_valid, y_pred)

r2 = r2_score(y_valid, y_pred)

print('MAE:{}, coeff of determination: {}'.format(score,r2))
# Preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test) # Your code here



# Check your answer

step_2.check()
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)