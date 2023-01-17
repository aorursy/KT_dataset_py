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
data=pd.read_csv("/kaggle/input/used-cars-data-pakistan/OLX_Car_Data_CSV.csv",encoding = "ISO-8859-1")

data.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(20,10))

sns.countplot(data=data,x="Brand")
sns.countplot(data=data,x="Transaction Type")
sns.relplot(data=data,x="KMs Driven",y="Price")
sns.lmplot(data=data,x="KMs Driven",y="Price")
data["-km"]=-1*data["KMs Driven"]

sns.lmplot(data=data,x="-km",y="Price")
dataMore100=data[data["Price"]>30000 & (data["Price"]<31000)]

sns.lmplot(data=dataMore100,x="KMs Driven",y="Price")
dataMore100=data[(data["Price"]>30000) & (data["Price"]<100000)]

sns.lmplot(data=dataMore100,x="KMs Driven",y="Price")
dataMore100=data[(data["Price"]>30000) & (data["Price"]<100000)]

sns.relplot(data=dataMore100,x="KMs Driven",y="Price",hue="Brand")
print(min(data["Price"]))

print(max(data["Price"]))
data[data["Price"]==max(data["Price"])]
data[data["Price"]==min(data["Price"])]
dataMore100=data[(data["KMs Driven"]>30000) & (data["KMs Driven"]<34000)]

sns.relplot(data=dataMore100,x="KMs Driven",y="Price",hue="Brand")
dataMore100=data[(data["KMs Driven"]>=0) & (data["KMs Driven"]<5000)]

sns.relplot(data=dataMore100,x="KMs Driven",y="Price",hue="Brand")
data.head()
from sklearn import preprocessing

from sklearn.impute import SimpleImputer

dataClean = data.dropna() 

encodedData= pd.DataFrame()

encodedData["Year"]=dataClean["Year"]

encodedData["km"]=dataClean["KMs Driven"]

encodedData["Price"]=dataClean["Price"]

changeColumns=["Brand","Condition","Fuel","Model"]

for col in changeColumns:

    lr = preprocessing.LabelEncoder()

    temp=dataClean[col]

    encodedData[col]=lr.fit_transform(temp)
encodedData.isnull().sum()
x=encodedData[["Year","km","Brand","Condition","Fuel","Model"]]

y=encodedData["Price"]
from sklearn.datasets import load_boston

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

cross_val_score(regressor,x,y, cv=10)
from sklearn.ensemble import RandomForestRegressor

cross_val_score(RandomForestRegressor(),x,y, cv=10)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
rfr=RandomForestRegressor()

rfr.fit(x_train,y_train)

ypred=rfr.predict(x_test)

from sklearn.metrics import max_error

print(max_error(y_test,ypred))
y_test2=list(y_test)

for i in range(0,len(ypred)):

    if i==100:

        break

    print(int(ypred[i]),"    ",y_test2[i])
from sklearn.metrics import r2_score

r2_score(y_test, ypred)
rfr.feature_importances_
encodedData.columns
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x_train,y_train)

ypred=regressor.predict(x_test)

from sklearn.metrics import max_error

print(max_error(y_test,ypred))
r2_score(y_test, ypred)
regressor.feature_importances_
import xgboost as xgb

regresyon=xgb.XGBRegressor()

regresyon.fit(x_train,y_train)

ypred=regresyon.predict(x_test)

print(max_error(y_test,ypred))
r2_score(y_test, ypred)
regresyon=xgb.XGBRegressor(n_estimators = 40,learning_rate=0.01)

regresyon.fit(x_train,y_train)

ypred=regresyon.predict(x_test)

print(max_error(y_test,ypred))

print(r2_score(y_test, ypred))
regresyon=xgb.XGBRegressor(n_estimators = 5,learning_rate=0.1)

regresyon.fit(x_train,y_train)

ypred=regresyon.predict(x_test)

print(max_error(y_test,ypred))

print(r2_score(y_test, ypred))
import xgboost as xgb

regresyon=xgb.XGBRegressor()

regresyon.fit(x_train,y_train)

ypred=regresyon.predict(x_test)

print(max_error(y_test,ypred))

print(regresyon.feature_importances_)
x.columns
from sklearn.dummy import DummyRegressor

regresyon=DummyRegressor()

regresyon.fit(x_train,y_train)

ypred=regresyon.predict(x_test)

print(max_error(y_test,ypred))

print(r2_score(y_test, ypred))