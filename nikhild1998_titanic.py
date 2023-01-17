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
titanic_data=pd.read_csv("../input/train.csv")
print(titanic_data.describe())
print(titanic_data.head(10))
print(titanic_data.columns.values)
titanic_data
independent_values=['Pclass','Sex','Age','SibSp','Parch']
dependent_value=['Survived']
X=titanic_data[independent_values]
y=titanic_data[dependent_value]
from sklearn.preprocessing import OneHotEncoder
#onehotencoder=OneHotEncoder(categorical_features=[1])
#X=onehotencoder.fit_transform(X).toarray()
X=pd.get_dummies(X)
from sklearn.impute import SimpleImputer
impute=SimpleImputer()
X=impute.fit_transform(X)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=0)
'''print(X_train.head())
print(X_test.head())
print(y_train.head())'''
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
fitted_data=model.fit(X_train,y_train)
predict_data=model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae1=mean_absolute_error(y_test,predict_data)
print("The mean absolute error is:",mae1)
from xgboost import  XGBClassifier
model_xgb= XGBClassifier(n_estimators=1000)
model_xgb.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_test,y_test)],verbose=False)
predict_xgb=model_xgb.predict(X_test)
mae2=mean_absolute_error(y_test,predict_xgb)
print("The mean absolute error is(xgb):",mae2)