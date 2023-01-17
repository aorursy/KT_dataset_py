# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import time

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/weatherAUS.csv")
data.head()
data = data.drop(columns=['RISK_MM'],axis=1)
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
df_num_col = ["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"]

data_num=data[df_num_col]

imputer=imputer.fit(data_num)

data[df_num_col]=imputer.transform(data_num)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

df_cat_col = ["WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow","Date","Location"]
data_cat=data[df_cat_col].fillna('NA')
for i in range(len(data_cat.columns)):

  data_cat.iloc[:,i] = labelencoder.fit_transform(data_cat.iloc[:,i])

  

data[df_cat_col]=data_cat
x=data.iloc[:,0:22].values

y=data.iloc[:,22].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression

import time

from sklearn.metrics import accuracy_score

t0=time.time()

logreg=LogisticRegression(random_state=0)

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

score = accuracy_score(y_test,y_pred)

print('Logistic Regression Accuracy :',score)

print('Logistic Regression Time taken :' , time.time()-t0)
from sklearn.tree import DecisionTreeClassifier

t0=time.time()

destree=DecisionTreeClassifier(random_state=0)

destree.fit(x_train,y_train)

y_pred=destree.predict(x_test)

score = accuracy_score(y_test,y_pred)

print('Decision Tree Accuracy :',score)

print('Decision Tree Time taken :' , time.time()-t0)
from sklearn.ensemble import RandomForestClassifier

t0=time.time()

rantree=RandomForestClassifier(random_state=0)

rantree.fit(x_train,y_train)

y_pred=rantree.predict(x_test)

score = accuracy_score(y_test,y_pred)

print('Random Tree Accuracy :',score)

print('Random Tree Time taken :' , time.time()-t0)