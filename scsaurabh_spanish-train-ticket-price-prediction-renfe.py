import numpy as np 

import pandas as pd 

import seaborn as sns

import datetime

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor
data = pd.read_csv('../input/renfe.csv', index_col=0)

data.head()
data.info()
data.isnull().sum()
data['price'].fillna(data['price'].mean(),inplace=True)
data.dropna(inplace=True)
data.drop('insert_date',axis=1,inplace=True)
data.isnull().sum()
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(data['origin'])

plt.show()
fig,ax = plt.subplots(figsize=(15,5))

ax = sns.countplot(data['destination'])

plt.show()
fig,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(data['train_type'])

plt.show()
fig,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(data['train_class'])

plt.show()
fig,ax = plt.subplots(figsize=(15,6))

ax = sns.countplot(data['fare'])

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.distplot(data['price'],rug=True)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='train_class',y='price',data=data)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x='train_type',y='price',data=data)

plt.show()
data = data.reset_index()
datetimeFormat = '%Y-%m-%d %H:%M:%S'

def fun(a,b):

    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)

    return(diff.seconds/3600.0)

    
data['travel_time_in_hrs'] = data.apply(lambda x:fun(x['start_date'],x['end_date']),axis=1) 
data.drop(['start_date','end_date'],axis=1,inplace=True)

data.head()
df1 = data[(data['origin']=="MADRID") & (data['destination']=="SEVILLA")]

df1.head()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x="train_type",y="price",data=df1)

plt.show()
df1 = data[(data['origin']=="MADRID") & (data['destination']=="BARCELONA")]

df1.head()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x="train_type",y="price",data=df1)

plt.show()
df1 = data[(data['origin']=="MADRID") & (data['destination']=="VALENCIA")]

df1.head()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x="train_type",y="price",data=df1)

plt.show()
df1 = data[(data['origin']=="MADRID") & (data['destination']=="PONFERRADA")]

df1.head()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)

plt.show()
f,ax = plt.subplots(figsize=(15,6))

ax = sns.boxplot(x="train_type",y="price",data=df1)

plt.show()
lab_en = LabelEncoder()

data.iloc[:,1] = lab_en.fit_transform(data.iloc[:,1])

data.iloc[:,2] = lab_en.fit_transform(data.iloc[:,2])

data.iloc[:,3] = lab_en.fit_transform(data.iloc[:,3])

data.iloc[:,5] = lab_en.fit_transform(data.iloc[:,5])

data.iloc[:,6] = lab_en.fit_transform(data.iloc[:,6])
data.head()
X = data.iloc[:,[1,2,3,5,6,7]].values

Y = data.iloc[:,4].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
lr = LinearRegression()

lr.fit(X_train,Y_train)
lr.score(X_test,Y_test)
lg = LGBMRegressor(n_estimators=1000)

lg.fit(X_train,Y_train)
lg.score(X_test,Y_test)