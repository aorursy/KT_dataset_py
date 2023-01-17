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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
df=pd.read_csv("../input/insurance/insurance.csv")

df
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
dfle=df

dfle.sex=le.fit_transform(dfle.sex)

dfle
dfle.charges=le.fit_transform(dfle.charges)

dfle.region=le.fit_transform(dfle.region)

dfle.smoker=le.fit_transform(dfle.smoker)

dfle
%matplotlib inline

plt.xlabel('age')

plt.ylabel('charges')

plt.scatter(dfle.age,dfle.charges,color='red',marker='+')
x=dfle.drop(['age','sex','bmi','children','smoker','region'],axis='columns')

x
y=dfle.charges

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
reg = linear_model.LinearRegression()

reg.fit(x,y)
reg.score(x_test,y_test)
from sklearn.tree import DecisionTreeRegressor

dec_regressor= DecisionTreeRegressor(criterion='mse',random_state=0)

dec_regressor.fit(x_train,y_train)

score=dec_regressor.score(x_test,y_test)

print("Decesion Tree Regression Accuracy score is ", score*100)
from sklearn.ensemble import RandomForestRegressor

rand_regressor= RandomForestRegressor(n_estimators=10,random_state=0)

rand_regressor.fit(x_train,y_train)

score=rand_regressor.score(x_test,y_test)

print("Random Forest Regression Accuracy score is ", score*100)