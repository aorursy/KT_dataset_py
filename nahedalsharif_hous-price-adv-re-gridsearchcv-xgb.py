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

parameters = {

    'n_estimators': list(range(100, 1001, 100)), 

    'learning_rate': [l / 100 for l in range(5, 100, 10)], 

    'max_depth': list(range(6, 70, 10))

}

parameters
my_randome_state=1486
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

gsearch = GridSearchCV(estimator=XGBRegressor(random_state=my_randome_state),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X_num, y)
best_n_estimators = gsearch.best_params_.get('n_estimators')

best_n_estimators
best_learning_rate = gsearch.best_params_.get('learning_rate')

best_learning_rate
best_max_depth = gsearch.best_params_.get('max_depth')

best_max_depth
final_model = XGBRegressor(n_estimators=best_n_estimators, 

                          learning_rate=best_learning_rate, 

                          max_depth=best_max_depth)
final_model.fit(X_num, y)
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', 

                     index_col='Id')

X_test 
X_test_num = X_test.select_dtypes(exclude=['object'])
preds_test = final_model.predict(X_test_num)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output
output.to_csv('submission.csv', index=False)

print('done!')