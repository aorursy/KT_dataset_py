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
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')

# remove the rows with missing target (SalePrice)
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

#separate target from predictors
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# take some validation set from training data - eg. 80 - 20 % split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Select categorical columns with relatively low cardinality 
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 100 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only - numerical and low cardinality categorical columns
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
# lets look how the training set looks like
X_train.head()
X_train.tail()
## do a little pre-processing for missing data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data - a simple imputation strategy
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# combine the preprocessing of numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
## define a gradient boosting model
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
# note that you can fine tune the parameters of xgb using grid search or other equivalent approaches
model_xgb = XGBRegressor(n_estimators=4000, learning_rate=0.01, n_jobs=4, random_state=0,  early_stopping_rounds=40, subsample = 0.4, reg_lambda = 0.9,
                       feature_selector = 'shuffle', tree_method = 'exact',
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
## lets do the training / fitting of the model here
# lets create a pipeline for combining the preprocessing and modeling
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_xgb)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train) 


# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
## generating test predictions
preds_test = my_pipeline.predict(X_test)
# Save the test predictions 
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

## submit your predictions - at the time of writing, it was in the top 4% of the leaderboard
# thanks to the mini course on ML: https://www.kaggle.com/learn/intermediate-machine-learning 
