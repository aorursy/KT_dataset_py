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
import pandas as pd
audi=pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')

audi.isnull().sum()
from sklearn.preprocessing import LabelEncoder

lc=LabelEncoder()
audi.model=lc.fit_transform(audi.model)
audi.transmission=lc.fit_transform(audi.transmission)
audi.fuelType=lc.fit_transform(audi.fuelType)

X=audi.drop('price',axis=1)
y=audi.price
columns=X[['model','transmission','fuelType']]
columns.columns
l=pd.get_dummies(data=columns,columns=columns.columns)
l.head(3)
audi['model'].unique()
X.drop(['model','transmission','fuelType'],axis=1,inplace=True)
X_=pd.concat((X,l),axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X_,y,test_size=.25)
LiR=LinearRegression()
LiR.fit(X_train,y_train)
pred1=LiR.predict(X_test)

DcT=DecisionTreeRegressor()
DcT.fit(X_train,y_train)
pred2=DcT.predict(X_test)


svr=SVR()
svr.fit(X_train,y_train)
pred3=svr.predict(X_test)


naive=GaussianNB()
naive.fit(X_train,y_train)
pred4=naive.predict(X_test)


knn=KNeighborsRegressor()
knn.fit(X_train,y_train)
pred5=knn.predict(X_test)


RFR=RandomForestRegressor()
RFR.fit(X_train,y_train)
pred6=RFR.predict(X_test)



gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
pred7=gbr.predict(X_test)


xgb=XGBRegressor()
xgb.fit(X_train,y_train)
pred8=xgb.predict(X_test)


lgb=lgb.LGBMRegressor()
lgb.fit(X_train,y_train)
pred9=lgb.predict(X_test)


cat=CatBoostRegressor()
cat.fit(X_train,y_train)
pred10=cat.predict(X_test)

print('r2 Score of LinearRegression Model: ', r2_score(y_test,pred1))
print('r2 Score of DecisionTreeRegressor Model: ', r2_score(y_test,pred2))
print('r2 Score of SVR Model: ', r2_score(y_test,pred3))
print('r2 Score of GaussianNB Model: ', r2_score(y_test,pred4))
print('r2 Score of KNeighborsRegressor Model: ', r2_score(y_test,pred5))
print('r2 Score of RandomForestRegressor Model: ', r2_score(y_test,pred6))
print('r2 Score of GradientBoostingRegressor Model: ', r2_score(y_test,pred7))
print('r2 Score of XGBRegressor Model: ', r2_score(y_test,pred8))
print('r2 Score of LGBMRegressor Model: ', r2_score(y_test,pred9))
print('r2 Score of CatBoostRegressor Model: ', r2_score(y_test,pred10))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_=scaler.fit_transform(X_)

X_train, X_test, y_train, y_test = train_test_split(X_,y,test_size=.25)
import lightgbm as lgb

LiR=LinearRegression()
LiR.fit(X_train,y_train)
pred1=LiR.predict(X_test)

DcT=DecisionTreeRegressor()
DcT.fit(X_train,y_train)
pred2=DcT.predict(X_test)


svr=SVR()
svr.fit(X_train,y_train)
pred3=svr.predict(X_test)


naive=GaussianNB()
naive.fit(X_train,y_train)
pred4=naive.predict(X_test)


knn=KNeighborsRegressor()
knn.fit(X_train,y_train)
pred5=knn.predict(X_test)


RFR=RandomForestRegressor()
RFR.fit(X_train,y_train)
pred6=RFR.predict(X_test)



gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
pred7=gbr.predict(X_test)


xgb=XGBRegressor()
xgb.fit(X_train,y_train)
pred8=xgb.predict(X_test)


lgb=lgb.LGBMRegressor()
lgb.fit(X_train,y_train)
pred9=lgb.predict(X_test)


cat=CatBoostRegressor()
cat.fit(X_train,y_train)
pred10=cat.predict(X_test)

print('r2 Score of LinearRegression Model: ', r2_score(y_test,pred1))
print('r2 Score of DecisionTreeRegressor Model: ', r2_score(y_test,pred2))
print('r2 Score of SVR Model: ', r2_score(y_test,pred3))
print('r2 Score of GaussianNB Model: ', r2_score(y_test,pred4))
print('r2 Score of KNeighborsRegressor Model: ', r2_score(y_test,pred5))
print('r2 Score of RandomForestRegressor Model: ', r2_score(y_test,pred6))
print('r2 Score of GradientBoostingRegressor Model: ', r2_score(y_test,pred7))
print('r2 Score of XGBRegressor Model: ', r2_score(y_test,pred8))
print('r2 Score of LGBMRegressor Model: ', r2_score(y_test,pred9))
print('r2 Score of CatBoostRegressor Model: ', r2_score(y_test,pred10))

