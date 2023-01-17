# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor  #forest model for prediction

from sklearn.compose import ColumnTransformer  #to transfere columns from type to another

from sklearn.impute import SimpleImputer  # for missing values

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = "/kaggle/input/human-resources-data-set/HRDataset_v13.csv" # the data file direction

data_all = pd.read_csv(data_dir)
data_all.head()
data_all.columns
XY_features = ['ManagerID','PerfScoreID','PositionID']  # all used features

XY = data_all[XY_features]

XY.head()
XY = XY.dropna(axis = 'index', how = 'any') #droping rows with missing values
features_columns = ['ManagerID','PositionID']

X = XY[features_columns]

Y = XY.PerfScoreID



X.head(400)

#number = LabelEncoder()

#X['Sex'] = number.fit_transform(X['Sex'].astype('str'))

#X.Sex = X.Sex * 1.0

#X.head(400)
X.head(100)
Y.head(400)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,train_size = 0.8, test_size = 0.2, random_state = 3)



my_model = RandomForestRegressor( random_state=0)

my_model.fit(X_train, Y_train)

preds = my_model.predict(X_valid)

print(mean_absolute_error(preds, Y_valid))
from xgboost import XGBRegressor





my_model2 = XGBRegressor(n_estimators =100, learning_rate = 0.0490)

my_model2.fit(X_train, Y_train)



pred2 = my_model2.predict(X_valid)

print(mean_absolute_error(Y_valid, pred2))
from sklearn.tree import DecisionTreeRegressor



my_model3 = DecisionTreeRegressor()

my_model3.fit(X_train,Y_train)



pred3 = my_model3.predict(X_valid)

print(mean_absolute_error(Y_valid, pred3))