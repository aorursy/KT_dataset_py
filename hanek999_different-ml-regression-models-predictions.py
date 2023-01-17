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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
data = pd.read_csv('../input/insurance/insurance.csv')

data.head(5)
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['sex']= le.fit_transform(data['sex']) 
data['smoker']=le.fit_transform(data['smoker']) 
data.head(5)
X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()

regressor.fit(X_poly, y_train)
y_pred1 = regressor.predict(poly_reg.transform(X_test))

np.set_printoptions(precision=2)

print(np.concatenate((y_pred1.reshape(len(y_pred1),1), y_test.reshape(len(y_test),1)),1))
r2_score(y_test, y_pred1)
from sklearn.tree import DecisionTreeRegressor

regress = DecisionTreeRegressor(random_state = 0)

regress.fit(X_train, y_train)
y_pred2 = regress.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))
r2_score(y_test, y_pred2)
from sklearn.ensemble import RandomForestRegressor

regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor1.fit(X_train, y_train)
y_pred3 = regressor1.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred3.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))
r2_score(y_test, y_pred3)
y = y.reshape(len(y),1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

y_train = sc_y.fit_transform(y_train)
from sklearn.svm import SVR

regressor2 = SVR(kernel = 'rbf')

regressor2.fit(X_train, y_train)
y_pred4 = sc_y.inverse_transform(regressor2.predict(sc_X.transform(X_test)))

np.set_printoptions(precision=2)

print(np.concatenate((y_pred4.reshape(len(y_pred4),1), y_test.reshape(len(y_test),1)),1))
r2_score(y_test, y_pred4)