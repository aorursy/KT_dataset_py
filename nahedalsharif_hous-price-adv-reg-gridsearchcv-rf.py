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
filepath=('../input/house-prices-advanced-regression-techniques/train.csv')

df=pd.read_csv(filepath,index_col='Id')

df
y=df.SalePrice

X=df.drop('SalePrice',axis=1)
X
y
X_num = X.select_dtypes(exclude=['object'])

X_num
for col in X_num.columns:

    if X_num[col].isna().sum() > 0:

        print(col, X_num[col].isna().sum()   / len(X_num) )

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num))
parameters = {

    'n_estimators': list(range(100, 1001, 100)), 

    'max_leaf_nodes': list(range(2, 50, 5)), 

    'max_depth': list(range(6, 70, 5))

}

parameters
my_randome_state=1486
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

gsearch = GridSearchCV(estimator=RandomForestRegressor(random_state=my_randome_state),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X_num_imputed, y)
best_n_estimators = gsearch.best_params_.get('n_estimators')

best_n_estimators
best_max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes')

best_max_leaf_nodes
best_max_depth = gsearch.best_params_.get('max_depth')

best_max_depth
final_model = RandomForestRegressor(n_estimators=best_n_estimators, random_state=my_randome_state, 

                          max_leaf_nodes=best_max_leaf_nodes, 

                          max_depth=best_max_depth)
final_model.fit(X_num_imputed, y)
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', 

                     index_col='Id')

X_test 
X_test_num = X_test.select_dtypes(exclude=['object'])
for col in X_test_num.columns:

    if X_test_num[col].isna().sum() > 0:

        print(col, X_test_num[col].isna().sum()   / len(X_test_num) )
X_test_num_imputed = pd.DataFrame(imputer.transform(X_test_num))

X_test_num_imputed.columns = X_test_num.columns

X_test_num_imputed
preds_test = final_model.predict(X_test_num_imputed)
len(preds_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output
output.to_csv('submission.csv', index=False)

print('done!')