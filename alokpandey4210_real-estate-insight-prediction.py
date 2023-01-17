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
df=pd.read_csv('/kaggle/input/real-estate-price-prediction/Real estate.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['No','Y house price of unit area'],axis=1)
y=df['Y house price of unit area']
X.isnull().sum()
X.hist(bins=50)
X.boxplot(column='X1 transaction date',by='X2 house age')
X['X1 transaction date'].plot('density')
X['X2 house age'].plot('density',color='Red')
X.boxplot(column='X3 distance to the nearest MRT station',by='X2 house age')
X['X3 distance to the nearest MRT station'].plot('density',color='Yellow')
X['X4 number of convenience stores'].plot('density')
X.boxplot(column='X4 number of convenience stores',by='X3 distance to the nearest MRT station')
X['X5 latitude'].plot('density',color='Blue')
X.boxplot(column='X5 latitude',by='X4 number of convenience stores')
X['X6 longitude'].plot('density',color='Black')
X.boxplot(column='X6 longitude',by='X5 latitude')
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.preprocessing import MinMaxScaler

scalerX=MinMaxScaler(feature_range=(0,1))

X_train[X_train.columns]=scalerX.fit_transform(X_train[X_train.columns])

X_test[X_test.columns]=scalerX.transform(X_test[X_test.columns])
from sklearn.tree import DecisionTreeRegressor

pz=DecisionTreeRegressor(max_depth=3)

pz.fit(X_train,y_train)

pz.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor

qm=RandomForestRegressor()

qm.fit(X_train,y_train)

qm.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

wc=LinearRegression()

wc.fit(X_train,y_train)

wc.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesRegressor

oa=ExtraTreesRegressor()

oa.fit(X_train,y_train)

oa.score(X_test,y_test)
from sklearn.linear_model import SGDRegressor

pb=SGDRegressor()

pb.fit(X_train,y_train)

pb.score(X_test,y_test)
from sklearn.linear_model import RANSACRegressor

qm=RANSACRegressor()

qm.fit(X_train,y_train)

qm.score(X_test,y_test)

#save model

import pickle

file_name='Estate.sav'

tuples=(oa,X)

pickle.dump(tuples,open(file_name,'wb'))
f=wc.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error

print(mean_squared_error(y_test,f))

print(mean_absolute_error(y_test,f))
print(y_test[0:5],' ',f[0:5])