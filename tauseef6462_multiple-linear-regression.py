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
dataset = pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')

dataset
dataset = dataset.drop(['car_ID'],axis=1)

dataset
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

print(X)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3,4,5,6,7,13,14,16,])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
type(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(y_test, y_pred)

RMSE = MSE**0.5
print(MSE)

print(RMSE)
y_range = y_test.max() - y_test.min()
error_margin = error_margin = (RMSE / y_range) * 100
r2_score = r2_score(y_test, y_pred)
print(error_margin)

print(r2_score)