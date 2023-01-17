import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X.columns if  X[cname].dtype != "object"]



# Columns that nan represent that the item does not exist

importent_nan = {'Condition2', 'Exterior2nd', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'}

importent_nan = set(categorical_cols).intersection(importent_nan)

unimp_nan = set(categorical_cols) - importent_nan



# Keep selected columns only

my_cols = categorical_cols + numerical_cols
def get_model(numeric_imput, learning_rate, n_estimators):

    # Preprocessing for numerical data

    numerical_transformer = ColumnTransformer(transformers=[

        ('GarageYrBlt', SimpleImputer(strategy='constant', fill_value=1899), ['GarageYrBlt']),

        ('others', SimpleImputer(strategy=numeric_imput), list(set(numerical_cols) - set(['GarageYrBlt'])))

    ])

    

    categorical_imputer = ColumnTransformer(transformers=[

        ('importent_nan', SimpleImputer(strategy='constant', fill_value='not_exist'), list(importent_nan)),

        ('others', SimpleImputer(strategy='most_frequent'), list(unimp_nan))

    ])

    

                                              

    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[

        ('imputer', categorical_imputer),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])

    my_cols = categorical_cols + numerical_cols

    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('cat', categorical_transformer, categorical_cols)

        ])

    

    # Define model

    model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0,)

    

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                  ('model', model)

                                 ])

    return my_pipeline
model = get_model('mean', 0.08, 110)

model.fit(X, y)

pred = model.predict(X_test)
# Save test predictions to file

output = pd.read_csv('../input/sample_submission.csv')

output['SalePrice'] = pred

output.to_csv('submission.csv', index=False)