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
house_filepath='../input/house-prices-advanced-regression-techniques/train.csv'

df = pd.read_csv(house_filepath)
df
df.columns
for col in df.columns:

    print(col, len(df[col].unique()))
for col in df.columns:

    print(col,(df[col].dtypes))

    #print(df.dtypes)
df = df.select_dtypes(exclude=['object'])

df
X=df.dropna(axis=0, subset=['SalePrice'])

X
y=df.SalePrice
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)
parameters = [{

    'n_estimators': list(range(200, 701, 100)), 

    'learning_rate': [x / 100 for x in range(5, 100, 10)], 

    'random_state': list(range(0, 20, 2))

}]

print(parameters)
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5, verbose=7)
gsearch.fit(X, y)

best_n_estimators = gsearch.best_params_.get('n_estimators')

print('best_n_estimators=',best_n_estimators)

best_learning_rate = gsearch.best_params_.get('learning_rate')

print('learning_rate=',best_learning_rate)

best_random_state = gsearch.best_params_.get('random_state')

print('best_random_state=',best_random_state)

final_model = XGBRegressor(n_estimators=best_n_estimators, 

                          learning_rate=best_learning_rate, 

                          random_state=best_random_state)
final_model.fit(X, y)
df_test_filepath='../input/house-prices-advanced-regression-techniques/test.csv'

X_test=pd.read_csv(df_test_filepath)
X_test = X_test.select_dtypes(exclude=['object'])

preds_test = final_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)