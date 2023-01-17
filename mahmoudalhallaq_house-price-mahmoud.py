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
from xgboost.sklearn import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV





train_data_file = '../input/home-data-for-ml-course/train.csv'

test_data_file = '../input/home-data-for-ml-course/test.csv'



home_data = pd.read_csv(train_data_file)

test_data = pd.read_csv(test_data_file)

home_data
home_data.info()
home_data.isnull().sum()
diff_col= set(test_data)-set(home_data)

home_data.drop(columns=diff_col,axis = 1, inplace=True)

test_data.drop(columns=diff_col,axis = 1, inplace=True)
y = home_data.SalePrice
filteredColumns = home_data.dtypes[home_data.dtypes == np.object]

listOfColumnNames = list(filteredColumns.index)

listOfColumnNames

home_data.drop(listOfColumnNames, axis=1,inplace=True)

test_data.drop(listOfColumnNames, axis=1,inplace=True)
X = home_data.select_dtypes(exclude='object')

X = X.drop(columns=['SalePrice'])

X
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
parameters = [{

'n_estimators': list(range(100, 300, 100)), 

'learning_rate': [x / 100 for x in range(5, 101, 5)],

'random_state' : range(5, 101, 5),

}]
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5)



gsearch.fit(train_X, train_y)



gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate'),gsearch.best_params_.get('max_depth')
from sklearn.metrics import mean_absolute_error

f_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

f_model.fit(train_X, train_y)

pred = f_model.predict(val_X)

mean_absolute_error(val_y,pred)
fin_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'),learning_rate = gsearch.best_params_.get('learning_rate'),

                       max_depth =gsearch.best_params_.get('max_depth'),random_state = 1 )

fin_model.fit(X, y)

pred = fin_model.predict(test_data)
test_out = pd.DataFrame({

    'Id': test_data.index, 

    'SalePrice': pred,

})
test_out.to_csv('submission.csv', index=False)