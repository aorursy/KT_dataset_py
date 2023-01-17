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
# read the data
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv')
print(X_full.columns)
X_test.index
#dropping rows with missing target
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)
# selecting categorical variables with low cardinality
low_car_col = [col for col in X_full.columns if X_full[col].nunique()<10 and X_full[col].dtype == "object"]
numerical_col = [col for col in X_full.columns if X_full[col].dtype in ['int64', 'float64']]
selected_col = low_car_col + numerical_col

X = X_full[selected_col].copy()
X_test = X_test_full[selected_col].copy()

X_test.index
#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_preprocess = SimpleImputer(strategy='median')
cat_preprocess = Pipeline(steps=[('imp', SimpleImputer(strategy='constant')),
                                 ('enc', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_preprocess, numerical_col),
    ('cat', cat_preprocess, low_car_col)
])
def score(n_estimators, preprocessor):
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, random_state=0)
    my_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    score = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return score.mean()

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
n_estimators_candidate = range(800, 1201, 50)

dict = {n : score(n, preprocessor) for n in n_estimators_candidate}
dict
best_model = XGBRegressor(n_estimators=850, learning_rate=0.05, random_state=0)
my_pipeline = Pipeline(steps=[
    ('preprocess',preprocessor),
    ('model', best_model)
])
my_pipeline.fit(X, y)
preds_test = my_pipeline.predict(X_test)
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index+1461,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
output