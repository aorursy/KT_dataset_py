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
data_path = '../input/home-data-for-ml-course/train.csv'
X=pd.read_csv(train_data_path)
X.columns
X.describe()
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X[cols_with_missing].count()
# Filter out the cols which has more than 50% missing values
cols_to_droped_missing = [col for col in cols_with_missing if X[col].count() <  730]
X[cols_to_droped_missing].count()
from xgboost import XGBRegressor
#from sklearn.metrics import mean_absolute_error
model = XGBRegressor()

#1st Step: Drop the categrocial columns 
object_cols = [col for col in X.columns if X[col].dtype == object]
#list(set(object_cols+cols_to_droped_missing))
selected_X = X.drop(list(set(object_cols+cols_to_droped_missing)), axis=1)
selected_y = selected_X.SalePrice
selected_X = selected_X.drop(["SalePrice"], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(selected_X, selected_y, random_state = 0)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(imputed_X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(imputed_X_valid, y_valid)], 
             verbose=False)
test_data_path = '../input/home-data-for-ml-course/test.csv'
X_test=pd.read_csv(test_data_path)

selected_X_test = X_test.drop(list(set(object_cols+cols_to_droped_missing)), axis=1)
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(selected_X_test))
#selected_X_test[61:90]

predict_y = my_model.predict(imputed_X_test)



output = pd.DataFrame({'Id': X_test.Id,
                      'SalePrice': predict_y})
output.to_csv('house_submission.csv', index=False)