# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
SampleSubmission = pd.read_csv("../input/SampleSubmission.csv")

data = pd.read_csv("../input/train.csv",index_col='Square_ID')

print(data.shape)
cols=data.columns

print("The columns in the data are:\n")

print(cols)
data.describe()
data.isnull().sum()
predict_cols=['X','Y','elevation','precip 2014-11-16 - 2014-11-23',

       'precip 2014-11-23 - 2014-11-30', 'precip 2014-11-30 - 2014-12-07',

       'precip 2014-12-07 - 2014-12-14', 'precip 2014-12-14 - 2014-12-21',

       'precip 2014-12-21 - 2014-12-28', 'precip 2014-12-28 - 2015-01-04',

       'precip 2015-01-04 - 2015-01-11', 'precip 2015-01-11 - 2015-01-18',

       'precip 2015-01-18 - 2015-01-25', 'precip 2015-01-25 - 2015-02-01',

       'precip 2015-02-01 - 2015-02-08', 'precip 2015-02-08 - 2015-02-15',

       'precip 2015-02-15 - 2015-02-22', 'precip 2015-02-22 - 2015-03-01',

       'precip 2015-03-01 - 2015-03-08', 'precip 2015-03-08 - 2015-03-15','LC_Type1_mode']

label_cols=['target_2015']

X=data.loc[:,predict_cols]

y=data.loc[:,label_cols]

print("Shape of X is:",X.shape)

print("Shape of y is:",y.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

rf=RandomForestRegressor()

cv_score=-1*cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')

print("Mean squared error is:",cv_score.mean())
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y)

print("Train shapes:",X_train.shape,"-----",y_train.shape)

print("Validation shapes:",X_valid.shape,"-----",y_valid.shape)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_valid, y_valid)],verbose=True)