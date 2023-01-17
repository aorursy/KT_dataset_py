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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



data = pd.read_csv('../input/train.csv')

data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = data.SalePrice

X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])





my_imputer = Imputer()

Imp_X = my_imputer.fit_transform(X)

#Imp_y = my_imputer.fit_transform(y)

#test_X = my_imputer.transform(test_X)

train_X, test_X, train_y, test_y = train_test_split(Imp_X, y, test_size=0.25)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, verbose=False)
# make predictions

predictions = my_model.predict(test_X)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
my_model_1 = XGBRegressor(n_estimators=1000)

my_model_1.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
# make predictions

predictions_1 = my_model_1.predict(test_X)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_1, test_y)))
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model_2.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
# make predictions

predictions_2 = my_model_2.predict(test_X)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_2, test_y)))
from sklearn.ensemble import GradientBoostingRegressor

my_model_3 = GradientBoostingRegressor()

# fit the model as usual

my_model_3.fit(train_X, train_y)

predictions_3 = my_model_3.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_3, test_y)))
from sklearn.tree import DecisionTreeRegressor

model_dt = DecisionTreeRegressor(random_state=1)

# Fit Model

model_dt.fit(train_X, train_y)

predictions_dt = model_dt.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_dt, test_y)))
model_dt1 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

model_dt1.fit(train_X, train_y)

predictions_dt1 = model_dt1.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions_dt1, test_y)))