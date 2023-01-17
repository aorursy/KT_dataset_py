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
df=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/train.csv.zip")

df.head()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns
df.columns
df=df.drop('Id',axis=1)
df.columns

df['Open Date']=pd.to_datetime(df['Open Date'])

df
df['month']=[x.month for x in df['Open Date']]

df['year']=[x.year for x in df['Open Date']]
df

sns.countplot(df['month'])
df.groupby('month')['revenue'].mean()
sns.barplot('month','revenue',data=df)
df=df.drop('Open Date',axis=1)

df['Type'].value_counts()

ty={'FC':0,'IL':1,'DT':2}

df['Type']=df['Type'].map(ty)
df.info()
df['City Group'].value_counts()
cg={'Big Cities':0,'Other':1}

df['City Group']=df['City Group'].map(cg)
df['City'].value_counts()
x=0

c={}

for i in df['City'].unique():

    c.update({i:x})

    x=x+1

c
df['City']=df['City'].map(c)
from sklearn.model_selection import train_test_split

x=df.drop('revenue',axis=1)

y=df['revenue']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error ,mean_squared_error,r2_score
dr=DecisionTreeRegressor()

dr=dr.fit(X_train,Y_train)

pred=dr.predict(X_test)

s=mean_absolute_error(Y_test,pred)

s1=mean_squared_error(Y_test,pred)

s2=r2_score(Y_test,pred)

print("The MAE with the DecisionTreeRegressor is: "+str(s))

print("The MsE with the DecisionTreeRegressor is: "+str(s1))

print("The R2_Score with the DecisionTreeRegressor is: "+str(s2))

lr=LinearRegression()

lr=lr.fit(X_train,Y_train)

pred=lr.predict(X_test)

s=mean_absolute_error(Y_test,pred)

s1=mean_squared_error(Y_test,pred)

s2=r2_score(Y_test,pred)

print("The MAE with the linear regressor is: "+str(s))

print("The MsE with the linear regressor is: "+str(s1))

print("The R2_Score with the linear regressor is: "+str(s2))
r=RandomForestRegressor()

r=r.fit(X_train,Y_train)

pred=r.predict(X_test)

s=mean_absolute_error(Y_test,pred)

s1=mean_squared_error(Y_test,pred)

s2=r2_score(Y_test,pred)

print("The MAE with the RandomForestRegressor is: "+str(s))

print("The MsE with the RandomForestRegressor is: "+str(s1))

print("The R2_Score with the RandomForestRegressor is: "+str(s2))
k=KNeighborsRegressor()

k=k.fit(X_train,Y_train)

pred=k.predict(X_test)

s=mean_absolute_error(Y_test,pred)

s1=mean_squared_error(Y_test,pred)

s2=r2_score(Y_test,pred)

print("The MAE with the KNeighborsRegressor is: "+str(s))

print("The MsE with the KNeighborsRegressor is: "+str(s1))

print("The R2_Score with the KNeighborsRegressor is: "+str(s2))

x=XGBRegressor()

x=dr.fit(X_train,Y_train)

pred=x.predict(X_test)

s=mean_absolute_error(Y_test,pred)

s1=mean_squared_error(Y_test,pred)

s2=r2_score(Y_test,pred)

print("The MAE with the XGBRegressor is: "+str(s))

print("The MsE with the XGBRegressor is: "+str(s1))

print("The R2_Score with the XGBRegressor is: "+str(s2))
df_t=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/test.csv.zip")

df_t.head()

i_d=df_t['Id']

df_t=df_t.drop('Id',axis=1)
df_t['Open Date']=pd.to_datetime(df_t['Open Date'])
df_t['month']=[x.month for x in df_t['Open Date']]

df_t['year']=[x.year for x in df_t['Open Date']]
df_t=df_t.drop('Open Date',axis=1)

df_t['Type'].value_counts()

ty={'FC':0,'IL':1,'DT':2}

df_t['Type']=df_t['Type'].map(ty)

cg={'Big Cities':0,'Other':1}

df_t['City Group']=df_t['City Group'].map(cg)

x=0

c={}

for i in df_t['City'].unique():

    c.update({i:x})

    x=x+1

df_t['City']=df_t['City'].map(c)
df_t.head()
df_t.dropna
df_t['Type']=df_t['Type'].fillna(0)
df_t.info()


p=k.predict(df_t)

sub=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv")

sub['Id']=i_d
sub.to_csv("Submission1.csv",index=False)