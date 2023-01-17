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
home_data_file = '../input/house-prices-advanced-regression-techniques/train.csv'

test_data_file = '../input/house-prices-advanced-regression-techniques/test.csv'
home_data = pd.read_csv(home_data_file,index_col='Id')

test_data = pd.read_csv(test_data_file,index_col='Id')
home_data.info()
home_data.isnull().sum()
diff_col= set(test_data)-set(home_data)

home_data.drop(columns=diff_col,axis = 1, inplace=True)

test_data.drop(columns=diff_col,axis = 1, inplace=True)
cols_with_miss_train_data = [col for col in home_data.columns 

                            if home_data[col].isnull().any()]



cols_with_miss_test_data = [col for col in test_data.columns 

                            if test_data[col].isnull().any()]



all_missing_columns = cols_with_miss_train_data + cols_with_miss_test_data

home_data.drop(columns=all_missing_columns, axis=1,inplace=True)

test_data.drop(columns=all_missing_columns, axis=1,inplace=True)
filteredColumns = home_data.dtypes[home_data.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

listOfColumnNames
home_data.drop(listOfColumnNames, axis=1,inplace=True)

test_data.drop(listOfColumnNames, axis=1,inplace=True)
target_col = 'SalePrice'

y = home_data[target_col]

y
X = home_data.select_dtypes(exclude='object')

X = X.drop(columns=[target_col])

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
from xgboost.sklearn import XGBRegressor

parameters = [{

'n_estimators': list(range(100, 301, 100)), 

'learning_rate': [x / 100 for x in range(5, 101, 5)],

'max_depth':[6,7,8]

}]

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5)



gsearch.fit(X_train, y_train)



gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate'),gsearch.best_params_.get('max_depth')
from sklearn.metrics import mean_absolute_error



f_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

f_model.fit(X_train, y_train)

pred = f_model.predict(X_train)



mean_absolute_error(y_train,pred)
final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

final_model.fit(X, y)

pred = final_model.predict(test_data)
test_out = pd.DataFrame({

    'Id': test_data.index, 

    'SalePrice': pred,

})
test_out.to_csv('submission.csv', index=False)