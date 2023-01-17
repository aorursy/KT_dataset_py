# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import io
print(os.listdir("../input"))

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

print(train_data.columns)
X = train_data.drop(['SalePrice'], axis = 1)
y = train_data.SalePrice


##gives you first 10 entries
#print(X.describe) 

##gives you dimensions
#print(X.shape)
#preprocss data so trings are turned into values
ohe_X = pd.get_dummies(X) 

#fill in value for missing values
#default strategy is mean
my_imputer = SimpleImputer()
imp_ohe_X = my_imputer.fit_transform(ohe_X)
#split into train/test
X_train, X_test, y_train, y_test = train_test_split(imp_ohe_X, 
                                                    y, 
                                                    train_size = 0.7, 
                                                    test_size = 0.3, 
                                                    random_state = 0)
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state = 1)
# Fit the model
tree_model.fit(X_train,y_train)
basic_predict_prices = tree_model.predict(X_test)
mean_absolute_error(y_test,basic_predict_prices)