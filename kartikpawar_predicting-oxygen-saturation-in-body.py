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
data1 = pd.read_csv("/kaggle/input/body-measurements/1520792014405.csv")

data2 = pd.read_csv("/kaggle/input/body-measurements/3560848988588.csv")
data1.shape
data2.shape
data1.isnull().any().sum()
data2.isnull().any().sum()
data = pd.concat([data1, data2])
data.shape
data
X = data.iloc[:, :-1]

y = data.iloc[:, -1]
print(X.shape)

print(y.shape)
X_train = X.iloc[:19889, :]
y_train = y.iloc[:19889]
X_test = X.iloc[19889:, :]
y_test = y.iloc[19889:]
print(X_train.shape)

print(X_test.shape)
import xgboost
clf = xgboost.XGBRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred)
y_test = y_test.values.reshape(-1, 1)
y_pred = np.reshape(y_pred, (-1, 1))
y_test[:5]
y_pred[:5]
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))