# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 



# Read csv files

test_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_full = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



# Get X and y from train

train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_full.SalePrice

X = train_full.drop(['SalePrice'], axis=1)



### Get columns required ###



# Number of missing values in each column of training data

missing_val_count_by_column = (X.isnull().sum())

cols_to_drop = []

for i in range(0,len(missing_val_count_by_column)):

    if missing_val_count_by_column.iloc[i]/len(X.index)*100 > 34:   # drop column if more than 34% null

        cols_to_drop.append(missing_val_count_by_column.index[i])

X.drop(cols_to_drop, axis=1, inplace=True)



# Carindality is number of unique values in the column

# We want to only on-hot encode low cardinality columns

low_cardinality_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() < 10]

high_cardinality_cols = [col for col in X.columns if X[col].dtype == 'object' and X[col].nunique() >= 10]

numeric_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]



my_cols = low_cardinality_cols + numeric_cols

X = X[my_cols]

X_test = test_full[my_cols]





### Preprocessing Pipeline ###



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

#from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer





numerical_transformer = SimpleImputer(strategy='median')

low_cardinality_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



#high_cardinality_transformer = Pipeline(steps=[

#    ('imputer', SimpleImputer(strategy='most_frequent')),

#    ('ordinal', OrdinalEncoder()),

#])



preprocessor = ColumnTransformer(transformers=[

    ('num', numerical_transformer, numeric_cols),

    ('cat1', low_cardinality_transformer, low_cardinality_cols),

    #('cat2', high_cardinality_transformer, high_cardinality_cols)

])



### Random Forest ###



from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score





def get_xgb_score(n_estimators):

    xgb_model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.10, random_state=1)

    my_pipeline = Pipeline(steps=[

        ('preprocessor', preprocessor),

        ('model', xgb_model)

    ])

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')

    avg_mae = scores.mean()

    return avg_mae



xgb_results = {}

for i in range(2,12):

    xgb_results[i*100] = get_xgb_score(i*100)



xgb_n_estimators_best = min(xgb_results, key=xgb_results.get)

xgb_model = XGBRegressor(xgb_n_estimators_best, learning_rate=0.05, random_state=0)

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', xgb_model)

                             ])

xgb_pipeline.fit(X, y)

xgb_preds_test = xgb_pipeline.predict(X_test)



### Output ###



output = pd.DataFrame({'Id': X_test['Id'],

                       'SalePrice': xgb_preds_test})

output.to_csv('submission.csv', index=False)