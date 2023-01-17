# General library imports

import numpy as np

import pandas as pd
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Read the data

X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)
# Indicate missing values

# Idea from https://www.kaggle.com/mrshih/here-s-a-house-salad-it-s-on-the-house

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

for col in cols_with_missing:

    X[col+'_was_missing']= X[col].isnull()

    X[col+'_was_missing']= X[col].isnull()



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X.columns if

                    X[cname].nunique() < 10 and 

                    X[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X.columns if 

                X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X = X[my_cols].copy()

X_test = X_test[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



numerical_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



model = XGBRegressor(random_state=0, 

                      learning_rate=0.005, n_estimators=1000,

                      max_depth=4,colsample_bytree=0.5, subsample=0.5,

                      min_child_weight = 1, gamma = 0, scale_pos_weight = 1)



pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                           ('model', model)

                          ])
from sklearn.model_selection import GridSearchCV



param_grid = {'model__n_estimators': np.arange(4400, 5500, 100),

              #'model__max_depth': np.arange(1, 11, 1),

              #'model__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],

              #'model__booster':['gbtree'],

              #'model__min_child_weight':[0, 1, 5, 10, 50],

              #'model__objective':['reg:squarederror'],

            #'model__gamma':[0.001, 0.01, 0.1, 1, 10, 100],

            #'model__max_delta_step':[0, 0.01, 0.1, 1, 10, 100]

            #'model__subsample':[0.4, 0.45, 0.5, 0.55, 0.6],

            #'model__colsample_bytree':[0.4, 0.45, 0.5, 0.55],

            #'model__colsample_bylevel':[0.25,0.5, 0.75, 1],

            #'model__colsample_bynode':[0.25,0.5, 0.75, 1]

            #'model__reg_lambda':[0, 0.01, 0.1, 1, 10, 100]

            #'model__reg_alpha':[0.001, 0.01, 0.1, 1, 10]

             }



#search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=8)

#search.fit(X, y)

#print("Best parameter (CV score=%0.3f):" % search.best_score_)

#print(search.best_params_)

#print(search.grid_scores_)
# score = 14907 @ n_estimators = 3460 (15172 @ 1000)

model = XGBRegressor(random_state=0, 

                      learning_rate=0.01, n_estimators=3460,

                      max_depth=4,colsample_bytree=0.5, subsample=0.5)





pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                           ('model', model)

                          ])



from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(pipeline, X, y,

                              cv=5, n_jobs=-1,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
pipeline.fit(X, y)

preds_test = pipeline.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)