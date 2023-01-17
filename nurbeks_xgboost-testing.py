# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_path = '../input/test.csv'
data = pd.read_csv(data_path)
data.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer


iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
home_data.dropna(axis=0, subset = ['SalePrice'], inplace=True)
y = home_data.SalePrice
X = home_data.drop(['SalePrice'], axis=1).select_dtypes(exclude = ['object'])


train_X, val_X, val_y, val_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size = 0.25)


my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import *

model = XGBRegressor(nthread = -1)
params = {'min_child_weight':[4, 5, 6], 'gamma':[i/100.0 for i in range(1,5)],  'subsample':[i/10.0 for i in range(1,6)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [4, 5, 6]}

clf = GridSearchCV(model, params)
clf.fit(train_X, train_y)
preds = clf.predict(val_X)
print("Mean Absolute Error: " + str(mean_absolute_error(preds, val_y)))
test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
test_X = test_data.select_dtypes(exclude=['object'])
test_X.head()
test_X = test_X.as_matrix()


test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

