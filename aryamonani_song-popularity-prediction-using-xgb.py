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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

sns.set()
X = pd.read_csv('../input/top-50/top50.csv',encoding='latin1') #encoding = 'latin1' because the data wouldnot be loaded into the variable otherwise due to decoding issues

X_test_full = pd.read_csv('../input/top-50/top50.csv',encoding='latin1')

X.describe()
X = X.drop(['Unnamed: 0'],axis=1)

X.describe(include='all')
y = X.Popularity
X.drop(['Popularity'], axis=1, inplace=True)
X.describe()
sns.distplot(y)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, 

                                                                          random_state = 0)
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10

                       and X_train_full[cname].dtype=='object']
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols 

X_train = X_train_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()
#one hot encoding for categorical data

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
model_1 = XGBRegressor()

model_1.fit(X_train, y_train)

predictions_1 = model_1.predict(X_valid)

mae_1 = mean_absolute_error(y_valid, predictions_1)

mae_1
model_2 = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)

model_2.fit(X_train, y_train)

predictions_2 = model_2.predict(X_valid)

mae_2 = mean_absolute_error(y_valid, predictions_2)

mae_2
model_3 = XGBRegressor(n_estimators = 10000, learning_rate = 0.05)

model_3.fit(X_train, y_train)

predictions_3 = model_3.predict(X_valid)

mae_3 = mean_absolute_error(y_valid, predictions_3)

mae_3
predictions_3
output = pd.DataFrame({'Track.Name' : y_valid.index,

                       'Popularity' : predictions_3})

output