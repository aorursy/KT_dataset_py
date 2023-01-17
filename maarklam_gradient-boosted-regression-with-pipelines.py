import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

import seaborn as sns



from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id') 

X_test = pd.read_csv('../input/test.csv', index_col='Id')



y = X.SalePrice

X.drop(columns=['SalePrice'], inplace=True)
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

numerical_cols = [col for col in X.columns if (X[col].dtype == 'int64' or X[col].dtype == 'float64')]
numerical_transformer = SimpleImputer()

categorical_transformer = Pipeline(steps=

                                   [('imputer', SimpleImputer(strategy='most_frequent')),

                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(transformers=

                                 [('num', numerical_transformer, numerical_cols), 

                                  ('cat', categorical_transformer, categorical_cols)])



model = XGBRegressor(random_state=0)



pipeline = Pipeline(steps=

                   [('preprocess', preprocessor),

                   ('model', model)])



grid = GridSearchCV(pipeline,  

                    param_grid={'model__n_estimators': [2000, 3000],

                                'model__learning_rate' : [0.01, 0.05],                                

                                'model__min_child_weight' : [0, 1]

                               },

                    cv = 10,

                    scoring = 'neg_mean_absolute_error')



grid.fit(X, y)
print(f"Best model parameters: {grid.best_params_}")

print(f"Best score: {-1 * grid.best_score_}")
# save test predictions to file

predictions = grid.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})

output.to_csv('submission.csv', index=False)