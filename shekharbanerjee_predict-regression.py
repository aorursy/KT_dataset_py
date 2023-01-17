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
df = pd.read_csv('../input/boston-house-prices/housing.csv',names = ['data'])
df.head()
d = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv']
a = list(range(14))

a = [str(i) for i in a]

a
df1 = pd.DataFrame(df,columns = a)

#df1 = df1.fillna(0)
df1['data'] = df.data

for i in a:

    df1[i] = df1['data'].apply(lambda x : x.split()[int(i)])
df1 = df1[a]

df1.columns = d
df1.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = df1[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']]

y = df1['medv']
linR = LinearRegression(normalize=True)
trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)
linR.fit(trainX,trainY)
#accuracy of linear regression

print(linR.score(testX,testY))
from sklearn.ensemble import AdaBoostRegressor



regr = AdaBoostRegressor(random_state=0, n_estimators=100)

regr.fit(trainX, trainY)



regr.score(testX,testY)
from sklearn import ensemble



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

xg_model = ensemble.GradientBoostingRegressor(**params)





xg_model.fit(trainX,trainY)
#accuracy of gradient boosting regression

print(xg_model.score(testX,testY))
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)

poly.fit_transform(trainX)

linR.fit(trainX,trainY)

print(linR.score(testX,testY))
dict = { }
for i in range(2,10):

    

    poly.fit_transform(trainX)

    linR.fit(trainX,trainY)

    dict.update({ i: linR.score(testX,testY)})
dict
trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)
features = set(X.columns)
from sklearn import ensemble as en



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

xg_model = en.GradientBoostingRegressor(**params)





xg_model.fit(trainX,trainY)
dict={ }
for feat in features:

    train = trainX[list(features - set(feat))]

    test = testX[list(features - set(feat))]

    xg_model.fit(train ,trainY)

    dict.update({ feat : xg_model.score(test,testY)})

    
dict
sorted_x =sorted(dict.items(), key=lambda x: x[1], reverse=True)
sorted_x
trainX , testX ,trainY, testY = train_test_split( X, y, test_size=0.33, random_state=42)
from sklearn import linear_model as lin

reg = lin.Ridge(alpha = 0.001)

reg.fit(trainX,trainY)
#accuracy of the ridge regression

print(reg.score(testX,testY))
from sklearn import linear_model as lin

las = lin.Lasso(alpha = 0.001)

las.fit(trainX,trainY)
#accuracy of the ridge regression

print(las.score(testX,testY))
from sklearn.linear_model import ElasticNet

regr = ElasticNet(alpha = 0.5)

regr.fit(trainX,trainY)

#accuracy of the ridge regression

print(regr.score(testX,testY))