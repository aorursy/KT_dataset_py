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
#lets do hyundi
hyundi_df = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv')
hyundi_df.head()
hyundi_df.shape
hyundi_df.model.unique()
hyundi_df.model.value_counts()
hyundi_df['years_old'] = 2020-hyundi_df['year']
hyundi_df.drop('year',axis=1,inplace=True)
hyundi_df.head()
hyundi_df.transmission.unique()
hyundi_df.fuelType.unique()
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(hyundi_df['tax(£)'],hyundi_df['price'])
sns.pairplot(hyundi_df)
sns.heatmap(hyundi_df.corr(),annot = True)
hyundi_df.isnull().any()
#Feature Engineering
hyundi_df.drop('model',axis=1,inplace=True)
hyundi_df.head()
hyundi_df = pd.get_dummies(hyundi_df,drop_first=True)
hyundi_df.head()
X = hyundi_df.iloc[:,1:]
y = hyundi_df.iloc[:,0:1]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit(X)
X = sc_X.transform(X)
X = pd.DataFrame(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
y = pd.DataFrame(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

lin_reg.score(X_test,y_test)
lin_reg.coef_
lin_reg.intercept_
from sklearn.preprocessing import PolynomialFeatures
regressor_poly = PolynomialFeatures(degree=2)
X_poly = regressor_poly.fit_transform(X_train)

regressor_lin_2 = LinearRegression()
regressor_lin_2.fit(X_poly,y_train)
y_pred_2 = regressor_lin_2.predict(regressor_poly.fit_transform(X_test))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_2))
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
y_pred_svr = regressor.predict(X_test)
#print(r2_score(y_test))
print(r2_score(y_test,y_pred_svr))
from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state=0)
regressor_tree.fit(X_train,y_train)
y_pred_tree = regressor_tree.predict(X_test)
print(r2_score(y_test,y_pred_tree))
from sklearn.ensemble import RandomForestRegressor
regressor_forest = RandomForestRegressor(n_estimators=300,random_state=0)
regressor_forest.fit(X_train,y_train)
y_pred_forest = regressor_forest.predict(X_test)
print(r2_score(y_test,y_pred_forest))