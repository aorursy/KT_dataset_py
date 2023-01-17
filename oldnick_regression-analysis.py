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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")
print("shape: ",data.shape)

print("-----------")

print(data.isnull().sum())

print("-----------")

print(data.info())
#see the first five rows of dataset.

data.head(5)
print(data.describe())
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = "?", strategy = "most_frequent")

data["horsepower"] = imp.fit_transform(data[["horsepower"]])





data["horsepower"] = data["horsepower"].astype(float)
data["origin"].unique()
sns.pairplot(data)
sns.countplot(x = data["cylinders"],data = data)
sns.countplot(x = data["origin"],data = data)
x = data.iloc[:,1:8].values

y = data.iloc[:,0].values

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("encoder",OneHotEncoder(categories = "auto"),[6])], remainder = "passthrough")

x = np.array(ct.fit_transform(x))
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.25,random_state = 0) 

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

x_train_scaled = sc_x.fit_transform(X_train)

x_test_scaled = sc_x.transform(X_test)

y_train_scaled = sc_y.fit(Y_train.reshape(-1,1))

y_train_scaled = sc_y.transform(Y_train.reshape(-1,1))

y_test_scaled = sc_y.transform(Y_test.reshape(-1,1))
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(x_train_scaled,y_train_scaled)

y_pred = sc_y.inverse_transform(linreg.predict(x_test_scaled))
from sklearn.metrics import r2_score

r2_score(Y_test,y_pred)
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(X_train,Y_train)

y_pred = tree.predict(X_test)
importance = tree.feature_importances_

for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v))

    
from sklearn.metrics import r2_score

r2_score(Y_test,y_pred)
from sklearn.svm import SVR

svr = SVR()

svr.fit(x_train_scaled,y_train_scaled)

y_pred = sc_y.inverse_transform(svr.predict(x_test_scaled))
from sklearn.metrics import r2_score

r2_score(Y_test,y_pred)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_train,Y_train)

y_pred = rfr.predict(X_test)
from sklearn.metrics import r2_score

r2_score(Y_test,y_pred)