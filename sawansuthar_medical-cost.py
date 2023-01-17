# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/insurance/insurance.csv')

print(dataset.head())

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder= LabelEncoder()

list_1 = [1,4,5]

from sklearn.compose import ColumnTransformer 

C = 0

for i in list_1:

    X.iloc[:, i] = labelencoder.fit_transform(X.iloc[:, i])   

print(X.head())

X = X.values

#for i in list1:

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), list_1)], remainder='passthrough') 

X = np.array(columnTransformer.fit_transform(X))

    #C = X[:,i+C].max()

    #print(C)

print(X[0:5,:])

print(X.shape)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

regressor_1 = RandomForestRegressor(n_estimators = 100,random_state=0)

regressor_1.fit(X_train,y_train)

regressor_1.score(X_train,y_train)
regressor.score(X_test,y_test)
import statsmodels.api as sm

X = np.append(arr=X,values = np.ones((1338,1)),axis=1)

list_2 = [0,1,2,3,4,5,6,7,8,9,10,11]

X_opt = X[:,list_2]

regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()

P_val = regressor_ols.pvalues

high_p_index = P_val.argmax()

p = P_val[high_p_index]

print(high_p_index)

print(P_val)


while p>0.05 :

    rt = list_2.pop(high_p_index)

    X_opt = X[:,list_2]

    regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()

    P_val = regressor_ols.pvalues

    high_p_index = (P_val.argmax()).astype(int)

    p = P_val[high_p_index]

print(X_opt)   