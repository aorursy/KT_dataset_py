# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')

print(data)
X = data.drop([ 'Species'], axis=1)

Y=data['Species']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y= le.fit_transform(Y)

print(Y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression

lin_regres=LinearRegression()

lin_regres.fit(x_train,y_train)

y_pred=lin_regres.predict(x_test)
import statsmodels.formula.api as sm 

X_l=X.values

r_ols=sm.OLS(endog = Y, exog =X_l).fit()

print(r_ols.summary())





print("R2 VALUE OF Linear reg:")

print(r2_score(y_test, lin_regres.predict(x_test)) )