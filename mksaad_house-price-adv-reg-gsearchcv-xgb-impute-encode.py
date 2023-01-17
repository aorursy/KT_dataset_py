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
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', 

                 index_col='Id')

df
y = df.SalePrice

X = df.drop(columns=['SalePrice'], axis=1)
cat_cols = [col for col in X.columns if X[col].dtype == 'object']

len(cat_cols)
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', 

                     index_col='Id')

X_test 
y
X
y.isna().sum()
bad_cols = [ col for col in cat_cols if set(X[col].unique()) != set(X_test[col].unique()) ]

bad_cols                                       
len(bad_cols)
X.drop(columns=bad_cols, axis=1, inplace=True)

X_test.drop(columns=bad_cols, axis=1, inplace=True)
#update cat_cols

cat_cols = [col for col in X.columns if X[col].dtype == 'object']

len(cat_cols)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='most_frequent')

X_imputed = pd.DataFrame(imputer.fit_transform(X))

X_imputed.columns = X.columns



X_test_imputed = pd.DataFrame(imputer.transform(X_test))

X_test_imputed.columns = X_test.columns
X_imputed
X_test_imputed
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X_imputed_label = X_imputed.copy()

X_test_imputed_label = X_test_imputed.copy()

for col in cat_cols:

    X_imputed_label[col] = label_encoder.fit_transform(X_imputed[col])

    X_test_imputed_label[col] = label_encoder.transform(X_test_imputed[col])
X_imputed_label
X_test_imputed_label
X_imputed_label = X_imputed_label.apply(pd.to_numeric) 

X_test_imputed_label = X_test_imputed_label.apply(pd.to_numeric) 
X_imputed_label.dtypes
X.shape
X_test.shape
parameters = {

    'n_estimators': list(range(100, 1001, 100)), 

    'learning_rate': [x / 100 for x in range(5, 100, 10)], 

    'max_depth': list(range(6, 70, 10))

}

parameters
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

gsearch = GridSearchCV(XGBRegressor(random_state=1),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X_imputed_label, y)
best_n_estimators = gsearch.best_params_.get('n_estimators')

best_n_estimators
best_learning_rate = gsearch.best_params_.get('learning_rate')

best_learning_rate
best_max_depth = gsearch.best_params_.get('max_depth')

best_max_depth
final_model = XGBRegressor(n_estimators=best_n_estimators, random_state=1,

                          learning_rate=best_learning_rate, 

                          max_depth=best_max_depth)
final_model.fit(X_imputed_label, y)
preds_test = final_model.predict(X_test_imputed_label)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output
output.to_csv('submission.csv', index=False)

print('done!')