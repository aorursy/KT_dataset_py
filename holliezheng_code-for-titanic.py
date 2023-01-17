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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn import model_selection
# save filepath to variable for easier access
titanic_train_path = '/kaggle/input/titanic/train.csv'
titanic_test_path = '/kaggle/input/titanic/test.csv'

# read the data and store data in DataFrame 
train_data = pd.read_csv(titanic_train_path, index_col='PassengerId') 
X_test_full = pd.read_csv(titanic_test_path, index_col='PassengerId') 

# drop the row if the predict value Survived is NULL
train_data.dropna(axis = 0, subset =['Survived'], inplace = True)

# get the feature set X and predict value set y
X = train_data.drop(['Survived'], axis = 1)
y = train_data['Survived']

#valid/training set split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# select categorical columns with cardinality less than 10
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]
print(categorical_cols)

# select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]
print (numerical_cols)

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
# Preprocessing for numerical data - will fill nan with median value 
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data - will fill nan with the most frequent value then do onehot encoding
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
'''
MODEL SELECTION : this code can be commented out once the best parameters is selected for the XGBoost model
'''
# Define the parameters
xgb_params = {
'learning_rate': [0.001,0.005,0.01],
'n_estimators': np.arange(0, 1500, 250).tolist(),
'max_depth': [3, 5, 7, 9],
'gamma': np.arange(0, 1.1, 0.2).tolist(),
'subsample': [0.5, 0.7, 1],
'colsample_bytree': [0.5, 0.7, 1]
 }

# Model selection
grid = model_selection.RandomizedSearchCV(XGBRegressor(), xgb_params, n_jobs = 4, cv=5)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', grid)
                     ])
# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid).round(0)

print('MAE:', mean_absolute_error(y_valid, preds))
print(grid.best_params_)
# Use XGB Regressor as the model
model = XGBRegressor(n_estimators = 1250, subsample = 0.7, max_depth = 9, learning_rate = 0.005, gamma =0.4, colsample_bytree = 0.5 )
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)
# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid).round(0)
print('MAE:', mean_absolute_error(y_valid, preds))
# Preprocessing of test data, fit model
preds_test = clf.predict(X_test).round(0) 

# Save test predictions to file
output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': preds_test})
output = output.astype({'Survived': 'int64'})
output.to_csv('submission.csv', index=False)
print(output)