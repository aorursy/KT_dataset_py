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

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from matplotlib import rcParams

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

import xgboost as xgb
sns.set(style='darkgrid')
train=pd.read_csv('../input/used-cars-price-prediction/train-data.csv')

test=pd.read_csv('../input/used-cars-price-prediction/test-data.csv')
pd.pandas.set_option('display.max_columns',None)
train.head()
test.head()
train.info()
train.shape
test.shape
train.isna().sum()
test.isna().sum()
train=train.drop(['New_Price'],axis=1)

test=test.drop(['New_Price'],axis=1)
train['Name'].unique()
print(len(train['Name'].unique()))
train['Name'].value_counts()
for i in range(train.shape[0]):

    train.at[i, 'Manufacturing_Name'] = train['Name'][i].split()[0]
train['Manufacturing_Name'].unique()
train=train.drop(['Name'],axis=1)

train.head()
train.groupby('Manufacturing_Name')['Unnamed: 0'].count()
for i in range(test.shape[0]):

    test.at[i, 'Manufacturing_Name'] = test['Name'][i].split()[0]

    

test=test.drop(['Name'],axis=1)

test.head()
train['Mileage']=train['Mileage'].str.extract(r'(\d+\.\d+)').astype(float)
train.groupby('Mileage')['Unnamed: 0'].count()
train['Mileage'][train['Mileage']==0.0]=np.nan
train['Mileage']=train['Mileage'].fillna(train['Mileage'].mode()[0])
train.isna().sum()
test['Mileage']=test['Mileage'].str.extract(r'(\d+\.\d+)').astype(float)

test.groupby('Mileage')['Unnamed: 0'].count()
train['Mileage'][train['Mileage']==0.0]=np.nan

train['Mileage']=train['Mileage'].fillna(train['Mileage'].mode()[0])
train.groupby('Engine')['Unnamed: 0'].count()
train['Engine']=train['Engine'].str.extract(r'(\d+)').astype(float)
train['Engine']=train['Engine'].fillna(train['Engine'].mode()[0])
test.groupby('Engine')['Unnamed: 0'].count()
test['Engine']=test['Engine'].str.extract(r'(\d+)').astype(float)

test['Engine']=test['Engine'].fillna(test['Engine'].mode()[0])
test.isna().sum()
train.groupby('Power')['Unnamed: 0'].count()
train['Power']=train['Power'].str.extract(r'(\d+\.\d+)').astype(float)

train['Power']=train['Power'].fillna(train['Power'].mode()[0])
test['Power']=test['Power'].str.extract(r'(\d+\.\d+)').astype(float)

test['Power']=test['Power'].fillna(test['Power'].mode()[0])
train.groupby('Seats')['Unnamed: 0'].count()
train['Seats'][train['Seats']==0.0]=np.nan

train['Seats']=train['Seats'].fillna(train['Seats'].mode()[0])
train.groupby('Seats')['Unnamed: 0'].count()
test['Seats']=test['Seats'].fillna(test['Seats'].mode()[0])
train.isna().sum()
train.isna().sum()
plt.figure(figsize=(15,10))

sns.heatmap(train.corr(),annot=True)
plt.figure(figsize=(15,10))

chart=sns.countplot(x='Location',data=train)

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.figure(figsize=(15,10))

sns.countplot(x='Year',data=train)
plt.figure(figsize=(10,6))

sns.countplot(x='Fuel_Type',data=train)
plt.figure(figsize=(10,6))

sns.countplot(x='Transmission',data=train)
plt.figure(figsize=(10,6))

sns.countplot(x='Owner_Type',data=train)
plt.figure(figsize=(10,6))

sns.countplot(x='Seats',data=train)
plt.figure(figsize=(15,10))

chart=sns.countplot(x='Manufacturing_Name',data=train)

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
px.histogram(x='Price',data_frame=train)
plt.figure(figsize=(15,10))

sns.countplot(x='Owner_Type',hue='Fuel_Type',data=train)
plt.figure(figsize=(15,10))

sns.countplot(x='Year',hue='Fuel_Type',data=train)
plt.figure(figsize=(15,10))

sns.countplot(x='Year',hue='Owner_Type',data=train)
px.bar(data_frame=train,x='Manufacturing_Name',y='Price')
train.head()
train_Location=pd.get_dummies(train.Location,drop_first=True)

train_Location.head(2)
test_Location=pd.get_dummies(test.Location,drop_first=True)

test_Location.head(2)
train['Fuel_Type'].value_counts()
test['Fuel_Type'].value_counts()
train['Fuel_Type']=train['Fuel_Type'].replace({'Electric':'LPG'})
train_Fuel_Type=pd.get_dummies(train.Fuel_Type,drop_first=True)

train_Fuel_Type.head(2)
test_Fuel_Type=pd.get_dummies(test.Fuel_Type,drop_first=True)

test_Fuel_Type.head(2)
Trans={'Automatic':0,'Manual':1 }

train['Transmission']=train['Transmission'].map(Trans)
Trans={'Automatic':0,'Manual':1 }

test['Transmission']=test['Transmission'].map(Trans)
own={'First':4, 'Second':3, 'Third':2, 'Fourth & Above':1 }

train['Owner_Type']=train['Owner_Type'].map(own)
own={'First':4, 'Second':3, 'Third':2, 'Fourth & Above':1 }

test['Owner_Type']=test['Owner_Type'].map(own)
train_Manuf=pd.get_dummies(train.Manufacturing_Name,drop_first=True)

train_Manuf.head(2)
test_Manuf=pd.get_dummies(test.Manufacturing_Name,drop_first=True)

test_Manuf.head(2)
train.head(2)
train=pd.concat([train,train_Location,train_Fuel_Type,train_Manuf],axis=1)

train=train.drop(['Unnamed: 0','Location','Fuel_Type','Manufacturing_Name'],axis=1)

train.head(2)
test=pd.concat([test,test_Location,test_Fuel_Type,test_Manuf],axis=1)

test=test.drop(['Unnamed: 0','Location','Fuel_Type','Manufacturing_Name'],axis=1)

test.head(2)
X=train.drop(['Price'],axis=1)

y=train['Price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=5)
knr=KNeighborsRegressor(n_neighbors=2)

knr.fit(X_train,y_train)

y_pred=knr.predict(X_test)
score=r2_score(y_test,y_pred)

score
regressor=RandomForestRegressor(n_estimators = 100, random_state = 15)

regressor.fit(X_train,y_train)

y_Pred=regressor.predict(X_test)
scores=r2_score(y_test,y_Pred)

scores
regressor1 = xgb.XGBRegressor(n_estimators=200,gamma=0,max_depth=4)



regressor1.fit(X_train,y_train)

y_Pred=regressor1.predict(X_test)
score1=r2_score(y_test,y_Pred)

score1