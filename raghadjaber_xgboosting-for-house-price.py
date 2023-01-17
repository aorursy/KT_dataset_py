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
data_filepath='../input/home-data-for-ml-course/train.csv'

df=pd.read_csv(data_filepath)
df
df.columns
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X= df[features]

y=df.SalePrice
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)
from xgboost import XGBRegressor

parameters = [{

'n_estimators': list(range(100, 1001, 100)), 

'learning_rate': [x / 100 for x in range(0, 20, 5)],

'random_state': list(range(0, 10, 2))

}]



from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=5)



my_model=gsearch.fit(X_train, y_train)



print(gsearch.best_params_.get('n_estimators'))

print(gsearch.best_params_.get('learning_rate'))

print(gsearch.best_params_.get('random_state'))
# Get predictions

predictions = my_model.predict(X_valid)
from sklearn.metrics import mean_absolute_error



# Calculate MAE

mae = mean_absolute_error(predictions, y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae)