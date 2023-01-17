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
df = pd.read_csv("../input/housing.csv")
df.head()
df.describe()
df.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop("MEDV",axis = 1),df["MEDV"],
                                                    test_size = 0.3, random_state = 42)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
np.mean((y_pred - y_test)**2)**0.5
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
np.mean((y_pred - y_test)**2)**0.5
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
np.mean((y_pred - y_test)**2)**0.5
from sklearn.ensemble import AdaBoostRegressor
ab = RandomForestRegressor()
ab.fit(X_train,y_train)
y_pred = ab.predict(X_test)
np.mean((y_pred - y_test)**2)**0.5
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPRegressor
nn_model = MLPRegressor(hidden_layer_sizes=(50)) 
nn_model.fit(X_train,y_train) 
y_pred = nn_model.predict(X_test)
np.mean((y_pred - y_test)**2)**0.5