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
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

features
stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

stores.head()
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

train.head(10)
train
stores.Type.value_counts()
df=pd.merge(train,features,on=['Store','Date','IsHoliday'],how='inner')
df.head()
df=pd.merge(df,stores,on='Store',how='inner')

df.head(10)
df.isna().mean()*100
from sklearn.impute import SimpleImputer



markdown=pd.DataFrame(SimpleImputer().fit_transform(df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]),columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

df = df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)

df=pd.concat([df,markdown],axis=1)

df.dtypes
df['Date']=pd.to_datetime(df['Date'])

df['year']=df['Date'].dt.year

df['month']=df['Date'].dt.month

df['day']=df['Date'].dt.day

del df['Date']
from sklearn.preprocessing import LabelEncoder



df['Type']=LabelEncoder().fit_transform(df['Type'])

df['IsHoliday']=LabelEncoder().fit_transform(df['IsHoliday'])

df.dtypes
df['Weekly_Sales'].plot.box()
df.columns
df[['Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Temperature',

       'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']].plot(kind='box',subplots=1,layout=(3,5),figsize=(14,12))
from scipy.stats import zscore 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
X = df.drop('Weekly_Sales',axis=1)

y = df['Weekly_Sales']

X_scaled = X.apply(zscore)



X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=.3,random_state=34)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score



lr = LinearRegression()

dt= DecisionTreeRegressor()

rf = RandomForestRegressor()

models = [lr,dt,rf]



for model in models:

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print(r2_score(y_test,y_pred))
(pd.DataFrame([X.columns,rf.feature_importances_],columns=['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',

       'Unemployment', 'Type', 'Size', 'MarkDown1', 'MarkDown2', 'MarkDown3',

       'MarkDown4', 'MarkDown5', 'year', 'month', 'day']).T).plot.bar()
pd.DataFrame([X.columns,rf.feature_importances_],columns=['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',

       'Unemployment', 'Type', 'Size', 'MarkDown1', 'MarkDown2', 'MarkDown3',

       'MarkDown4', 'MarkDown5', 'year', 'month', 'day']).T
x1 = X_scaled.drop(['IsHoliday','year','MarkDown5','MarkDown4','MarkDown1','MarkDown2'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(x1,y,test_size=.3,random_state=34)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

r2_score(y_test,y_pred)